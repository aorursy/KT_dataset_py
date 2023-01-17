# Version of the notebook (used for output file names)
version = 2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Conv2D, SpatialDropout2D, BatchNormalization,Activation
from keras.layers import MaxPooling2D, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras import losses
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import gc
def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_loss_core(y_true, y_pred)

def iou_bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + 5 * iou_loss(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def focal_loss(y_true, y_pred):
    gamma=0.5
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def competition_metric(true, pred): #any shape can go

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

# For threshold determination
def faster_iou_metric_batch(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)
def conv_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    pool = MaxPooling2D((2,2))(conv2)
    return pool, conv2  # returns the block output and the shortcut to use in the uppooling blocks

def middle_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    
    return conv2

def deconv_block(neurons, block_input, shortcut, bn=False, dropout=None):
    deconv = Conv2DTranspose(neurons, (3, 3), strides=(2, 2), padding="same")(block_input)
    uconv = concatenate([deconv, shortcut])
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='he_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='he_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
        
    return uconv
    
def build_model(start_neurons, bn=False, dropout=None):
    
    input_layer = Input((64, 64, 1))
    
    # 64 -> 32
    conv1, shortcut1 = conv_block(start_neurons, input_layer, bn, dropout)

    # 32 -> 16
    conv2, shortcut2 = conv_block(start_neurons * 2, conv1, bn, dropout)
    
    # 16 -> 8
    conv3, shortcut3 = conv_block(start_neurons * 4, conv2, bn, dropout)
    
    # 8 -> 4
    conv4, shortcut4 = conv_block(start_neurons * 8, conv3, bn, dropout)
    
    # Middle
    convm = middle_block(start_neurons * 16, conv4, bn, dropout)
    
    # 4 -> 8
    deconv4 = deconv_block(start_neurons * 8, convm, shortcut4, bn, dropout)
    
    # 8 -> 16
    deconv3 = deconv_block(start_neurons * 4, deconv4, shortcut3, bn, dropout)
    
    # 16 -> 32
    deconv2 = deconv_block(start_neurons * 2, deconv3, shortcut2, bn, dropout)
    
    # 32 -> 64
    deconv1 = deconv_block(start_neurons, deconv2, shortcut1, bn, dropout=False)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv1)
    
    model = Model(input_layer, output_layer)
    return model
%%time
DATA_DIR = '../input/tgs-eda/'
train = pd.read_hdf(DATA_DIR + 'tgs_data.h5', key='filtered_train')
train.head()
np.random.seed(42)

NUM_CROPS = 25    # Number of crops for every image
WIDTH = 64
HEIGHT = 64
IMAGE_PIXELS = WIDTH * HEIGHT
MAX_X = 37
MAX_Y = 37

def coverage(pixels):
    if pixels == 0:
        return 0
    else:
        percentage = pixels / IMAGE_PIXELS
        return np.ceil(percentage * 10).astype(np.uint)

def random_crop_params():
    x = np.random.randint(0, MAX_X)
    y = np.random.randint(0, MAX_Y)
    flip = np.random.choice(a=[False, True])
    intensity = np.random.normal(1,0.2)
    if intensity == 0:
        intensity = 0
    return x, y, flip, intensity

def crop(img, x, y, flip, intensity=1):
    random_img = img[x:x+WIDTH,y:y+WIDTH]
    if flip:
        random_img = np.fliplr(random_img)
    random_img = random_img * intensity
    return random_img.reshape(WIDTH, HEIGHT, 1)

imgs_aug = []
masks_aug = []
for idx in train.index:
    img = train.loc[idx]['images']
    mask = train.loc[idx]['masks']
    for i in range(NUM_CROPS):
        r = i // 5
        c = i % 5
        x, y, flip, intensity = random_crop_params()
        random_img = crop(img, x, y, flip, intensity)
        imgs_aug.append(random_img)
        random_mask = crop(mask, x, y, flip)
        masks_aug.append(random_mask)

data = pd.DataFrame({'images':imgs_aug, 'masks':masks_aug})
print(data.shape)

# Calculate coverage
data['pixels'] = data['masks'].map(lambda x: np.sum(x/255)).astype(np.int16)
data['coverage'] = data['pixels'].map(coverage).astype(np.float16)

data.describe()

del imgs_aug
del masks_aug
del train
# Plot coverage distribution
labels, counts = np.unique(data['coverage'], return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.grid(axis='y')
plt.show()
# Split the train data into actual train data and validation data
# train_test_split already shuffles data by default, so no need to do it

X = np.stack(data['images']) / 255
y = np.stack(data['masks']) / 255



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=data.coverage, random_state=42)

del data
del X
del y


# Build and save model in JSON format

json_filename = 'unet_salt_{}.json'.format(version)

model = build_model(start_neurons=8, bn=True, dropout=False)
model_json = model.to_json()
with open(json_filename, 'w') as json_file:
    json_file.write(model_json)
early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1)
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_competition_metric', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
optimizer = SGD(lr=0.1, momentum=0.8, nesterov=False)
#optimizer = Adam(lr=0.1)

model.compile(optimizer=optimizer, loss=iou_bce_loss, metrics=['accuracy', competition_metric])
history = model.fit(X_train, y_train, batch_size=64, validation_data = [X_val, y_val], 
                    epochs=50, callbacks=[checkpoint, reduce_lr])
# Restore the best model's weight
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
model.load_weights(weights_filename)
# Let's see how the model performs (last model after training, not the saved best one)

# On the train set
print('*** Best model on train set ***')
model.evaluate(X_train, y_train)

# On the validation set
print('*** Best model on val set ***')  
model.evaluate(X_val, y_val)
print(model.metrics_names)

hist = history.history

figure, ax = plt.subplots(1,3, figsize=(18,6))
print(ax.shape)

def plot_history(history, metric, title, ax):
    ax.plot(history[metric])
    ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')                
    
plot_history(hist, 'loss', 'LOSS', ax[0])
plot_history(hist, 'acc', 'ACCURACY', ax[1])
plot_history(hist, 'competition_metric', 'COMPETITION METRIC', ax[2])


# Save image for reports
history_df = pd.DataFrame(hist)
plt.savefig('history_unet_salt_{}.png'.format(version))
thresholds = np.linspace(0, 1, 50)
y_val_pred = model.predict(X_val)
ious = np.array([faster_iou_metric_batch(y_val, np.uint8(y_val_pred > threshold)) for threshold in tqdm_notebook(thresholds)])
best_th_index = ious.argmax()
best_th = thresholds[best_th_index]
best_iou = ious[best_th_index]
plt.plot(thresholds, ious)
plt.plot(best_th, best_iou, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(best_th, best_iou))
plt.legend()
# Save image for reports
plt.savefig('threshold_selection_{}.png'.format(version))