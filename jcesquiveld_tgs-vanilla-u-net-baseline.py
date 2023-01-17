# Version of the notebook (used for output file names)
version = 9
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
# Function to upsize images
def upsize(img):
    return resize(img, (128, 128, 1), mode='constant', preserve_range=True)

# Function to downsize images
def downsize(img):
    return resize(img, (101, 101, 1), mode='constant', preserve_range=True)

# Downsize predictions to size (101,101,1)
def downsize_preds(preds):
    preds_resized = []
    for i in range(len(preds)):
        preds_resized.append(np.squeeze(downsize(preds[i])))
    return np.array(preds_resized)

# Works much faster than my previous implementation RLenc
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    np.where: locates the positions in the array where a given condition holds true.
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

#def iou_loss_core(y_true,y_pred):
#    intersection = y_true * y_pred
#    notTrue = 1 - y_true
#    union = y_true + (notTrue * y_pred)
#    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_loss_core(y_true, y_pred)

def iou_bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + 3 * iou_loss(y_true, y_pred)

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


def conv_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    pool = MaxPooling2D((2,2))(conv2)
    return pool, conv2  # returns the block output and the shortcut to use in the uppooling blocks

def middle_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='glorot_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    
    return conv2

def deconv_block(neurons, block_input, shortcut, bn=False, dropout=None):
    deconv = Conv2DTranspose(neurons, (3, 3), strides=(2, 2), padding="same")(block_input)
    uconv = concatenate([deconv, shortcut])
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='glorot_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='glorot_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
        
    return uconv
    
def build_model(start_neurons, bn=False, dropout=None):
    
    input_layer = Input((128, 128, 1))
    
    # 128 -> 64
    conv1, shortcut1 = conv_block(start_neurons, input_layer, bn, dropout)

    # 64 -> 32
    conv2, shortcut2 = conv_block(start_neurons * 2, conv1, bn, dropout)
    
    # 32 -> 16
    conv3, shortcut3 = conv_block(start_neurons * 4, conv2, bn, dropout)
    
    # 16 -> 8
    conv4, shortcut4 = conv_block(start_neurons * 8, conv3, bn, dropout)
    
    # Middle
    convm = middle_block(start_neurons * 16, conv4, bn, dropout)
    
    # 8 -> 16
    deconv4 = deconv_block(start_neurons * 8, convm, shortcut4, bn, dropout)
    
    # 16 -> 32
    deconv3 = deconv_block(start_neurons * 4, deconv4, shortcut3, bn, dropout)
    
    # 32 -> 64
    deconv2 = deconv_block(start_neurons * 2, deconv3, shortcut2, bn, dropout)
    
    # 64 -> 128
    deconv1 = deconv_block(start_neurons, deconv2, shortcut1, bn, dropout)
    
    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv1)
    
    model = Model(input_layer, output_layer)
    return model
%%time
DATA_DIR = '../input/tgs-reading-data-and-storing-in-hf5/'
train = pd.read_hdf(DATA_DIR + 'tgs_salt.h5', key='train')
test = pd.read_hdf(DATA_DIR + 'tgs_salt.h5', key='test')
submission = pd.read_hdf(DATA_DIR + 'tgs_salt.h5', key='submission')
train.head()
# Resize images to (128,128,1) and clip pixel values to [0,1]
images_resized = train.images.map(upsize)
masks_resized = train.masks.map(upsize)
X = np.stack(images_resized) / 255
y = np.stack(masks_resized) / 255
X_aug = np.concatenate((X, [np.fliplr(img) for img in X]), axis=0)
y_aug = np.concatenate((y, [np.fliplr(img) for img in y]), axis=0)
# Split the train data into actual train data and validation data
# train_test_split already shuffles data by default, so no need to do it

X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.25, random_state=42)
# Build and save model in JSON format

json_filename = 'unet_salt_{}.json'.format(version)

model = build_model(start_neurons=8, bn=True, dropout=0.05)
model_json = model.to_json()
with open(json_filename, 'w') as json_file:
    json_file.write(model_json)
load_previous_weights = False
if (load_previous_weights):
    # Restore the model 
    weights_filename = 'unet_salt_weights_{}.h5'.format(version)
    model.load_weights(weights_filename)
early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1)
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
optimizer = SGD(lr=0.2, momentum=0.8, decay=0.001, nesterov=False)
#optimizer = Adam(lr=0.01)

model.compile(optimizer=optimizer, loss=iou_loss, metrics=['accuracy', competition_metric])
history = model.fit(X_train, y_train, batch_size=16, validation_data = [X_val, y_val], 
                    epochs=1, callbacks=[checkpoint])
# Let's see how the model performs (last model after training, not the saved best one)

# On the train set
print('*** Last model on train set ***')
model.evaluate(X_train, y_train)

# On the validation set
print('*** Last model on val set ***')  
model.evaluate(X_val, y_val)
print(model.metrics_names)

figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))

# summarize history for loss
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.grid(True)
ax1.set_title('LOSS')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# summarize history for accuracy
ax2.plot(history.history['acc'])
ax2.plot(history.history['val_acc'])
ax2.grid(True)
ax2.set_title('ACCURACY')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

# summarize history for competition metric
ax3.plot(history.history['competition_metric'])
ax3.plot(history.history['val_competition_metric'])
ax3.grid(True)
ax3.set_title('COMPETITION METRIC')
ax3.set_ylabel('competition metric')
ax3.set_xlabel('epoch')
ax3.legend(['train', 'validation'], loc='upper left')

# Save image for reports
plt.savefig('history_unet_salt_{}.png'.format(version))