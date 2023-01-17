import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/quickdraw-doodle-recognition"))
import matplotlib.pyplot as plt
import ast
import pandas as pd
import cv2
from skimage.io import imread, imshow
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, LSTM, Dropout, Flatten
from keras.models import Model, load_model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model
owls = pd.read_csv('../input/quickdraw-doodle-recognition/train_simplified/owl.csv')
recog_counts = owls['recognized'].value_counts()
owls = owls[owls.recognized]
owls['timestamp'] = pd.to_datetime(owls.timestamp)
owls = owls.sort_values(by='timestamp', ascending=False)
owls['drawing'] = owls['drawing'].apply(ast.literal_eval)
owls.head()
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, drawing in enumerate(owls.drawing[-100:]):
    ax = axs[i // n, i % n]
    for x, y in drawing:
        ax.plot(x, -np.array(y), lw=3)
    ax.axis('off')
fig.savefig('owls.png', dpi=200)
plt.show();
for x,y in owls.drawing[51970]:
    print("x:",x,"  y:",y)
country_counts = owls['countrycode'].value_counts()
pd.DataFrame(country_counts).head()
top_10_states = list(country_counts[:10].index)
owls_top_10 = owls[owls['countrycode'].isin(top_10_states)]
g = sns.catplot(x="countrycode", data=owls_top_10, kind="count",height=5, aspect=4)
sns.barplot(x=recog_counts.index, y=recog_counts)
word = []
count = []
count_recog = []
for f in os.listdir("../input/quickdraw-doodle-recognition/train_simplified"):
    df = pd.read_csv("../input/quickdraw-doodle-recognition/train_simplified/" + f)
    word.append(df['word'][0])
    count.append(df.shape[0])
    count_recog.append(len(df[df['recognized'] == True]))

summary = pd.DataFrame({'word':word,'count':count,'count_recog':count_recog})
summary.head()
summary_top = summary.sort_values(by='count')[-10:]
summary_top.head()
summary_less = pd.concat([summary.sort_values(by='count')[-10:],summary.sort_values(by='count')[:10]])
g = sns.catplot(x="word", y="count", data=summary_less, kind="bar",height=5, aspect=4)
CSV_DIR = '../input/doodle-detection-dataprep'
filename = CSV_DIR + '/train_k0.csv.gz'
train_sample = pd.read_csv(filename)
train_sample.head()
# lens = []
# CSV_DIR = '../input/doodle-detection-dataprep'
# for k in range(100):
#     filename = os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(k))
#     for df in pd.read_csv(filename):
#         lens.append(len(df))
BATCH_SIZE = 128
MAX_TRAIN_EPOCHS = 20
STEPS_PER_EPOCH = 900
NCSVS = 100
CSV_DIR = '../input/doodle-detection-dataprep'
BASE_SIZE = 256
size = 128
word_encoder = LabelEncoder()
categories = [word.split('.')[0] for word in os.listdir(os.path.join("../input/quickdraw-doodle-recognition/train_simplified/"))]
word_encoder.fit(categories)
print('words', len(word_encoder.classes_), '=>', ', '.join([x for x in word_encoder.classes_[:50]]))
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                    x[i, :, :, 1] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                    x[i, :, :, 2] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = to_categorical(word_encoder.transform(df["word"].values),num_classes=340).astype(np.int32)
                yield x, y
def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
        x[i, :, :, 1] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
        x[i, :, :, 2] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x
train_datagen = image_generator_xd(batchsize=BATCH_SIZE, ks=range(NCSVS - 1), size=size)
train_x, train_y = next(train_datagen)
print ('train x shape:{}'.format(train_x.shape))
print ('train y shape:{}'.format(train_y.shape))
print('train_x', train_x.dtype, train_x.min(), train_x.max())
print('train_y', train_y.dtype, train_y.min(), train_y.max())
valid_set = pd.read_csv(os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=1000)
valid_x = df_to_image_array_xd(valid_set, size)
valid_y = to_categorical(word_encoder.transform(valid_set["word"].values),num_classes=340).astype(np.int32)
print ('valid x shape:{}'.format(valid_x.shape))
print ('valid y shape:{}'.format(valid_y.shape))
print('valid_x', valid_x.dtype, valid_x.min(), valid_x.max())
print('valid_y', valid_y.dtype, valid_y.min(), valid_y.max())
fig, m_axs = plt.subplots(4,4, figsize = (8, 8))
rand_idxs = np.random.choice(range(train_x.shape[0]), size = 16, replace=False)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = train_x[c_id, :, :, 0]  
    c_ax.imshow(test_arr, cmap=plt.cm.gray)
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(train_y[c_id])])
def cnn(input_shape):
    input_img = Input(input_shape)
    conv0= Conv2D(256, (3, 3), activation='relu', padding='valid')(input_img) 
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1= Conv2D(128, (3, 3), activation='relu', padding='valid')(pool0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2= Conv2D(64, (3, 3), activation='relu', padding='valid')(pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='valid')(pool2) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 
    flat = Flatten()(pool3)
    dense1 = Dense(680, activation='relu')(flat)
    dense2 = Dense(len(word_encoder.classes_), activation = 'softmax')(dense1)
    
    model =  Model(inputs = input_img, outputs = dense2, name = 'Doodle_model')    
    return model

def MobileNetV2_model():
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128,128,3), pooling='avg')
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    output = Dense(340, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def vgg16_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(128,128,3), pooling='avg')
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    output = Dense(340, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def resnet_model():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(128,128,3), pooling='avg')
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    output = Dense(340, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def run_model(model):
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['categorical_accuracy', top_3_accuracy])
    
    checkpoint = ModelCheckpoint("model_weights.best.hdf5", monitor='val_top_3_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True, period=1)
    early = EarlyStopping(monitor="val_top_3_accuracy", mode="max", verbose=2,patience=8)
    callbacks_list = [checkpoint, early]
    
    loss_history = [model.fit_generator(train_datagen,epochs=15,steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_data=(valid_x, valid_y),callbacks=callbacks_list,workers=1)]
    model.load_weights("model_weights.best.hdf5")
    model.save('model.h5')

    return loss_history

def display_plots(loss_history):    
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    loss = np.concatenate([mh.history['loss'] for mh in loss_history])
    val_loss  = np.concatenate([mh.history['val_loss'] for mh in loss_history])
    
    train_accuracy = np.concatenate([mh.history['top_3_accuracy'] for mh in loss_history])
    test_accuracy = np.concatenate([mh.history['val_top_3_accuracy'] for mh in loss_history])
    print ('train accuray: {}'.format(max(train_accuracy)))
    print ('test accuray: {}'.format(max(test_accuracy)))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (30,10))

    ax1.plot(epochs,train_accuracy, epochs,test_accuracy)
    ax1.legend(['Training', 'Validation'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_title('accuracy train vs validation')

    ax2.plot(epochs,loss, epochs,val_loss)
    ax2.legend(['Training', 'Validation'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('loss train vs validation')

valid_set = pd.read_csv(os.path.join(CSV_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=16)
valid_x = df_to_image_array_xd(valid_set, size)
valid_y = to_categorical(word_encoder.transform(valid_set["word"].values),num_classes=340).astype(np.int32)

def validation_and_display(model):
    valid_img_label = model.predict(valid_x, verbose=True)
    top_3_pred_valid = [word_encoder.classes_[np.argsort(-1*c_pred)[:3]] for c_pred in valid_img_label]
    top_3_pred_valid = [' '.join([col.replace(' ', '_') for col in row]) for row in top_3_pred_valid]
    
    fig, m_axs = plt.subplots(4,4, figsize = (20, 20))
    rand_idxs = np.random.choice(range(valid_x.shape[0]), size = 16, replace=False)
    for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
        test_arr = valid_x[c_id, :, :, 0]
        c_ax.imshow(test_arr,cmap=plt.cm.gray)
        c_ax.axis('off')
        c_ax.set_title(('pred:'+top_3_pred_valid[c_id],'gt:'+valid_set["word"].iloc[c_id]))

model = cnn(input_shape = train_x.shape[1:])
loss_history = run_model(model)
display_plots(loss_history)
plot_model(model, to_file='model_cnn.png')
model = vgg16_model()
loss_history = run_model(model)
display_plots(loss_history)
plot_model(model, to_file='model_vgg16.png')
model = MobileNetV2_model()
loss_history = run_model(model)
display_plots(loss_history)
plot_model(model, to_file='model_mobilenet.png')
model = resnet_model()
loss_history = run_model(model)
display_plots(loss_history)
validation_and_display(model)