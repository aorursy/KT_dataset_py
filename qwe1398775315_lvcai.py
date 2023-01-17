# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from collections import Counter,defaultdict
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/guangdong_round1_train1_20180903"))

# Any results you write to the current directory are saved as output.
math.ceil(1.1)
PATH="../input/guangdong_round1_train1_20180903/guangdong_round1_train1_20180903"
files = os.listdir(PATH)
label = list([name[0:2]for name in files])
data = list(zip(label,[os.path.join(PATH,file) for file in files]))
pd.value_counts(label).plot.bar()
pd.value_counts(label)/250
data_dict = defaultdict(list)
for k,v in data:
    data_dict[k].append(v)
for key in data_dict.keys():
    print(key)
    images = data_dict[key][0:4]
    img_data = list([imread(image) for image in images])
    fig,axes=plt.subplots(nrows=1, ncols=4, figsize=(16,8))
    axes[0].imshow(img_data[0])
    axes[1].imshow(img_data[1])
    axes[2].imshow(img_data[2])
    axes[3].imshow(img_data[3])
    plt.show()
from keras.layers import BatchNormalization,Conv2D,Input,Concatenate,Dense,Dropout,MaxPool2D,AveragePooling2D,Activation,GlobalAveragePooling2D,Reshape,multiply
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l1_l2
from keras.models import Model,load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
def images(image_paths,shape) :
    for v in tqdm(np.asarray(image_paths).reshape(-1)):
        yield resize(image=imread(v),output_shape=shape,mode='constant')
label,image_paths = zip(*data)
ohe_label = pd.get_dummies(label)
ohe_label.head()
# rus = RandomUnderSampler(random_state=0)
# X_train, y_train = rus.fit_sample(np.asarray(image_paths).reshape(-1,1), label)
X_train,X_test,y_train,y_test = train_test_split(np.asarray(image_paths).reshape(-1,1),ohe_label,test_size=0.3)
x_train = np.asarray(list(images(X_train,(224,224))))
x_test = np.asarray(list(images(np.asarray(image_paths).reshape(-1,1),(224,224))))
plt.imshow(x_test[25])
plt.show()
def build_model(shape, target_size,k, kernel_initializer="he_normal"):
    def sub_dense_block(x, k, kernel_initializer="he_normal"):
        # conv1 = Conv2D(k, padding="same", kernel_size=(1, 1), kernel_initializer=kernel_initializer,
        #                kernel_regularizer=l1_l2(l1=l1, l2=l2))
        conv2 = Conv2D(k, padding="same", kernel_size=(7, 7),use_bias=False, kernel_initializer=kernel_initializer)
        with tf.name_scope("sub_dense_block"):
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = conv1(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = conv2(x)
        return x

    def dense_block(x, k, block_size, kernel_initializer="he_normal"):
        output_set = [x]
        yi = None
        with tf.name_scope("dense_block"):
            for i in range(block_size):
                if len(output_set) == 1:
                    xi = output_set[0]
                else:
                    xi = Concatenate()(output_set)
                yi = sub_dense_block(xi, k, kernel_initializer=kernel_initializer)
                output_set.append(yi)
        return yi
    def se_block(x,out_dim):
        squeeze = GlobalAveragePooling2D()(x)
        excitation = Dense(units=out_dim // 4,activation='relu')(squeeze)
        excitation = Dense(units=out_dim,activation='sigmoid')(excitation)
        excitation = Reshape((1,1,out_dim))(excitation)
        scale = multiply([x,excitation])
        return scale
    def transition_layer(x, k, kernel_initializer="he_normal"):
        conv1 = Conv2D(k, padding="same", kernel_size=(1, 1),use_bias=False, kernel_initializer=kernel_initializer)
        with tf.name_scope("transition_layer"):
            x = conv1(x)
            x = se_block(x,k)
            x = AveragePooling2D()(x)
        return x

    inp = Input(shape=shape)
    x = Conv2D(k, kernel_size=(3, 3), padding='same',use_bias=False, kernel_initializer=kernel_initializer)(inp)
    x = AveragePooling2D()(x)
    # dense block 1
    x = dense_block(x, k, 3, kernel_initializer)
    x = transition_layer(x, k, kernel_initializer)
    #dense block 2
#     x = dense_block(x, k, 3, kernel_initializer)
#     x = transition_layer(x, k, kernel_initializer)
#     # dense block 3
#     x = dense_block(x, k, 3, kernel_initializer, l1, l2)
#     x = transition_layer(x, k, kernel_initializer, l1, l2)
    # dense block 4
    x = dense_block(x, k, 3, kernel_initializer)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2*k, activation="relu",kernel_initializer=kernel_initializer)(x)
    y = Dense(units=target_size, activation="softmax",kernel_initializer=kernel_initializer)(x)
    return Model(inp, y)
model = build_model((224,224,3),4,32)
check_point = ModelCheckpoint("best_model1.hdf5", monitor = "val_loss", verbose = 1,save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
lr_reducer = ReduceLROnPlateau(factor=0.1,min_lr=1e-5,patience=2,monitor="val_loss")
callbacks = [check_point,early_stop,lr_reducer]
datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
    rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
# datagen.fit(np.concatenate([x_test]))
model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 0.001), metrics = ["accuracy"])
model.summary()
history = model.fit_generator(datagen.flow(x_test,ohe_label,4), epochs = 100,validation_data=(x_test,ohe_label), verbose = 1, callbacks = callbacks)
best_model = load_model("best_model1.hdf5")
best_model.summary()
best_model.evaluate(x_test, ohe_label)
def predict(index):
    print("index",index,"predict",ohe_label.columns[np.argmax(best_model.predict(datagen.standardize(np.expand_dims(x_test[index],0))))],"label",label[index])
for i in range(250):
    predict(i)
sns.distplot(best_model.get_layer('conv2d_17').get_weights()[0].reshape(-1),kde=False,label='trained')
sns.distplot(model.get_layer('conv2d_25').get_weights()[0].reshape(-1),kde=False,label='scratch')
plt.legend()
first_layer_model = Model(inputs=best_model.input,outputs=best_model.get_layer('conv2d_1').output)
def plot(layer,img_index):
    plt.imshow(x_test[img_index])
    plt.show()
    print(label[img_index])
    first_layer_model = Model(inputs=best_model.input,outputs=best_model.get_layer(layer).output)
    features=first_layer_model.predict(np.expand_dims(x_test[img_index],0))[0]
    nums = features.shape[2]
    for i in range(math.ceil(nums/4)):
        fig,axes=plt.subplots(nrows=1, ncols=4, figsize=(16,8))
        axes[0].imshow(features[:,:,i*4+0],cmap='Greys')
        axes[1].imshow(features[:,:,i*4+1],cmap='Greys')
        axes[2].imshow(features[:,:,i*4+2],cmap='Greys')
        axes[3].imshow(features[:,:,i*4+3],cmap='Greys')
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')
        plt.show()
x_test[228]
plot('conv2d_1',228)
plot('conv2d_71')
final_layer_model = Model(inputs=best_model.input,outputs=best_model.get_layer('global_average_pooling2d_19').output)
plt.imshow(x_test[4])
plt.show()
final_layer_model.predict(np.expand_dims(x_test[4],0))
plt.imshow(x_test[231])
plt.show()
final_layer_model.predict(np.expand_dims(x_test[231],0))
print(label[223])

best_model.predict(datagen.standardize(np.expand_dims(x_test[223],0)))
print(label[22])
best_model.predict(datagen.standardize(np.expand_dims(x_test[223],0)))
