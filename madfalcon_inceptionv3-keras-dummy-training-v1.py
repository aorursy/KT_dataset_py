cur_env = 'kaggle' # 'kaggle' or 'ubuntu' 둘중 하나 입력
import pandas as pd

import numpy as np

import cv2

import os

from pathlib import Path

import skimage.io



from sklearn.preprocessing import OneHotEncoder #One-hot 인코더

import keras.backend as K #케라스 버전 2.3.1

import random

from sklearn.model_selection import train_test_split

from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping

from matplotlib.pyplot import imshow
def quadratic_kappa_coefficient(y_true, y_pred):

    y_true = K.cast(y_true, "float32")

    n_classes = K.cast(y_pred.shape[-1], "float32")

    weights = K.arange(0, n_classes, dtype="float32") / (n_classes - 1)

    weights = (weights - K.expand_dims(weights, -1)) ** 2



    hist_true = K.sum(y_true, axis=0)

    hist_pred = K.sum(y_pred, axis=0)



    E = K.expand_dims(hist_true, axis=-1) * hist_pred

    E = E / K.sum(E, keepdims=False)



    O = K.transpose(K.transpose(y_true) @ y_pred)  # confusion matrix

    O = O / K.sum(O)



    num = weights * O

    den = weights * E



    QWK = (1 - K.sum(num) / K.sum(den))

    return QWK



def quadratic_kappa_loss(scale=2.0):

    def _quadratic_kappa_loss(y_true, y_pred):

        QWK = quadratic_kappa_coefficient(y_true, y_pred)

        loss = -K.log(K.sigmoid(scale * QWK))

        return loss

        

    return _quadratic_kappa_loss
from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions

from keras import models, Model

from keras.layers import Input,Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.losses import categorical_crossentropy
input_shape = (299, 299, 3)

if cur_env == 'ubuntu':

    base_net = InceptionResNetV2(weights='keras_pre_trained_model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)

    #base_net = VGG16(weights='keras_pre_trained_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)

else:

    base_net = InceptionV3(weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)

    #base_net = InceptionResNetV2(weights='../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)

    #base_net = VGG16(weights='../input/keras_pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)



for layer in base_net.layers:

    layer.trainable = False
model = models.Sequential()

model.add(base_net)



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(6, activation = "softmax"))

model.summary()
model = Model(inputs = model.input, outputs = model.output)
#loss = categorical_crossentropy,

model.compile(optimizer = Adam(lr=1e-3), loss = quadratic_kappa_loss(scale=6.0), \

             metrics = ['accuracy',quadratic_kappa_coefficient])
if cur_env == 'ubuntu':

    dir = 'dataset/'

    train_df = pd.read_csv(dir+'train.csv')

    test_df = pd.read_csv(dir+'test.csv')

    train_df['image_path'] = [dir + 'train_images/' +image_name +".tiff" for image_name in train_df['image_id']]

    test_df['image_path'] = [dir + 'train_images/' +image_name +".tiff" for image_name in test_df['image_id']]



else: #캐글일 경우

    HOME = Path("../input/prostate-cancer-grade-assessment")

    TRAIN = Path("train_images")

    CUSTOM = Path('../input/panda-conv-16x128x128/conv_train_images')

    train_df = pd.read_csv(str(HOME)+'/train.csv')

    test_df = pd.read_csv(str(HOME)+'/test.csv')

    train_df['image_path'] = [str(HOME/TRAIN/image_name) + ".tiff" for image_name in train_df['image_id']]

    train_df['conv_image_path'] = [str(CUSTOM/image_name) +".jpg" for image_name in train_df['image_id']]

    test_df['image_path'] = [str(HOME/TRAIN/image_name) + ".tiff" for image_name in test_df['image_id']]

    test_df['conv_image_path'] = [str(CUSTOM/image_name) +".jpg" for image_name in test_df['image_id']]



print(train_df.head(3))

print(test_df.head(3))

# train_x = train_df['image_id']

# train_y = train_df['isup_grade']

# test_x = test_df['image_id']

# print(f"train_x : {len(train_x)}, train_y : {len(train_y)}, test_x : {len(test_x)}")
# 'isup_grade'를 기준으로 라벨인코딩 진행

encoder = OneHotEncoder(handle_unknown = 'ignore')

encoder_labels = pd.DataFrame(encoder.fit_transform(train_df[['isup_grade']]).toarray())

#display(encoder_labels)



train_df = pd.merge(train_df, encoder_labels, left_index=True, right_index=True)

train_df.head(4)



train_df['conv_image_path'][0].endswith('.jpg')
# import matplotlib.pyplot as plt

# print(train_df['conv_image_path'][0])

# img = cv2.imread(train_df['conv_image_path'][0])



# print(img.shape)



# plt.imshow(img)

# plt.show()
# 이미지(tiff 파일) 호출 후 skimage를 통해 사이즈 축소 

#input_shape = (256, 256, 3) #모델에 넣을 사이즈



def get_image(image_location):

    #print(image_location)

    if image_location.endswith('.tiff'): #tiff 일 경우

        # 가장 작은 사이즈로 변환, 값은 -1, 0 ,1 ,2 ?

        image = skimage.io.MultiImage(image_location)

        image = image[-1]

    else: # jpg일경우

        image = cv2.imread(image_location)

    

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    # input 사이즈로 이미지 리사이즈

    image = cv2.resize(image, (input_shape[0], input_shape[1]))

    

    return image
# Function that shuffles annotation rows and chooses batch_size samples

#sequence = range(len(annotation_file))



def get_batch_ids(sequence, batch_size):

    sequence = list(sequence)

    random.shuffle(sequence)

    batch = random.sample(sequence, batch_size)

    return batch
# Basic data generator -> Next: add augmentation = False



def data_generator(data, batch_size):

    while True:

        data = data.reset_index(drop=True)

        indices = list(data.index)



        batch_ids = get_batch_ids(indices, batch_size)

        batch = data.iloc[batch_ids]['conv_image_path']



        X = [get_image(x) for x in batch]

        Y = data[[0, 1, 2, 3, 4, 5]].values[batch_ids]



        # Convert X and Y to arrays

        X = np.array(X)

        Y = np.array(Y)



        yield X, Y



# data: should be a pandas DF (train or val) obtained from train_test_split

# batch_size: is the size of the number of images passed through the net in one step
# Train -  Validation Split function

train, val = train_test_split(train_df, test_size = 0.3, random_state = 42)

display(train['conv_image_path'][1506])

#display(val.head(3))

print(len(train),len(val))

# import matplotlib.pyplot as plt

# print(train['conv_image_path'][1506])

# img = cv2.imread(train['conv_image_path'][1506])

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(img.shape)



# plt.imshow(img)

# plt.show()
# Some checkpoints

if cur_env == 'ubuntu':

    model_path = 'model_history/{epoch:02d}-{loss:.1f}-{val_loss:.1f}.h5'

else:

    model_path = './model.h5'



model_checkpoint = ModelCheckpoint(filepath=model_path, monitor = 'val_loss', verbose=0, save_best_only=True, save_weights_only=True)

early_stop = EarlyStopping(monitor='val_loss',patience=5,verbose=True)
EPOCHS = 50 

BS = 200



history = model.fit_generator(generator = data_generator(train, BS),

                              validation_data = data_generator(val, BS),

                              epochs = EPOCHS,

                              verbose = 1,

                              #steps_per_epoch = len(train)// BS,\

                              steps_per_epoch = 20,

                              validation_steps = 20, 

                              #validation_steps = len(val)// BS,\

                              callbacks =[model_checkpoint, early_stop])
import matplotlib.pyplot as plt

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
if cur_env == 'ubuntu': 

    sample_submission = pd.read_csv('dataset/sample_submission.csv')

else: #캐글일 경우

    sample_submission = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')

    TEST = Path("test_images")

    test_ann = pd.read_csv(HOME/'test.csv')
if os.path.exists(f'../input/prostate-cancer-grade-assessment/test_images/'):

    print('inference!')



    predictions = []

    for img_id in test_ann['image_id']:

        img = str(HOME/TEST/img_id) + ".tiff"

        print(img)

        image = get_image(img)

        image = image[np.newaxis,:]

        prediction = model.predict(image)

        # if we have 1 at multiple locations

        ind = np.where(prediction == np.amax(prediction))

        final_prediction = random.sample(list(ind[1]), 1)[0].astype(int)

        predictions.append(final_prediction)



    sample_submission = pd.DataFrame()

    sample_submission['image_id'] = test_ann['image_id']

    sample_submission['isup_grade'] = predictions

    sample_submission



    sample_submission.to_csv('submission.csv', index=False)

    sample_submission.head()

else:

    print('Test Images folder does not exist! Save the sample_submission.csv!')

    sample_submission.to_csv('submission.csv', index=False)