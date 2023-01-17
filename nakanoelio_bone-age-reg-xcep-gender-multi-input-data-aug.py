from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, concatenate, Conv2D
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import InputLayer, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
import seaborn as sns
%matplotlib inline
seed = 42
path = '../input/i2a2-bone-age-regression'
#!unzip -q '/content/gdrive/My Drive/Kaggle/I2A2_Bone_Regression/i2a2-bone-age-regression.zip' -d '/content/gdrive/My Drive/Kaggle/I2A2_Bone_Regression/'
train_df = pd.read_csv(path+'/train.csv')
test_df = pd.read_csv(path+'/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df = train_df.join(pd.get_dummies(train_df['patientSex'], prefix='patientSex'))
train_df.head()
plt.figure(figsize=(20, 10))

plt.subplot(232)
ax = sns.kdeplot(train_df['boneage'], label='Global Distribution')
ax = sns.kdeplot(train_df.loc[train_df.patientSex=='F']['boneage'], label='Female Distribution')
ax = sns.kdeplot(train_df.loc[train_df.patientSex=='M']['boneage'], label='Male Distribution')
ax.set_title('Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_xlabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(234)
ax = sns.distplot(train_df['boneage'])
ax.set_title('Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(235)
ax = sns.distplot(train_df.loc[train_df.patientSex=='F']['boneage'])
ax.set_title('Female - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(236)
ax = sns.distplot(train_df.loc[train_df.patientSex=='M']['boneage'])
ax.set_title('Male - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.show()
plt.figure(figsize=(15, 5))


plt.subplot(131)
ax = sns.boxplot(data=[train_df['boneage'],train_df.loc[train_df.patientSex=='F']['boneage'],train_df.loc[train_df.patientSex=='M']['boneage']])
ax.set_title('Bone Age Quantiles', color='b')
ax.set_xlabel('Data Set (0-Global, 1-Female, 2-Male)', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')
'''
plt.subplot(235)
ax = sns.distplot(train_df.loc[train_df.patientSex=='F']['boneage'])
ax.set_title('Female - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(236)
ax = sns.distplot(train_df.loc[train_df.patientSex=='M']['boneage'])
ax.set_title('Male - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')
'''
plt.show()
print('Statistics - Global - Bone Age')
train_df['boneage'].describe()
print('Statistics - Female - Bone Age')
train_df.loc[train_df.patientSex=='F']['boneage'].describe()
print('Statistics - Male - Bone Age')
train_df.loc[train_df.patientSex=='M']['boneage'].describe()
test_df[test_df['patientSex'] =='F'].shape
test_df[test_df['patientSex'] =='M'].shape
X_train, X_test, y_train, y_test = train_test_split(train_df[["fileName", "patientSex_F", "patientSex_M"]],train_df["boneage"], test_size = 0.15, random_state = seed)
X_train.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.15, random_state = seed)
X_train.shape

training_df = X_train.join(y_train)
validating_df = X_valid.join(y_valid)
training_df.head()
validating_df.head()
plt.figure(figsize=(20, 10))

plt.subplot(232)
ax = sns.kdeplot(y_train, label='y train')
ax = sns.kdeplot(y_valid, label='y valid')
ax = sns.kdeplot(y_test, label='y test')
ax.set_title('train/test/validation - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(234)
ax = sns.distplot(y_train)
ax.set_title('Y_train - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(235)
ax = sns.distplot(y_test)
ax.set_title('y_test -  - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.subplot(236)
ax = sns.distplot(y_valid)
ax.set_title('y_valid - Bone Age Distribution', color='b')
ax.set_xlabel('bone age', color='b')
ax.set_ylabel('frequency', color='b')
ax.tick_params(labelcolor='b')

plt.show()
img_size = (299)
batch_size = 10
patientSex_weight = 100

rotation_range = 20
width_shift_range = 0.1
height_shift_range = 0.1
horizontal_flip = True
shear_range = 0.1
zoom_range = [0.4,0.5]


datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode='nearest',rescale=1./255)#validation_split=0.25)

def gen_flow_for_two_inputs(mult_imp_df):
    data_iter=datagen.flow_from_dataframe(dataframe=mult_imp_df, 
                                          directory=path+'/images', 
                                          x_col='fileName', y_col=(['boneage','patientSex_F','patientSex_M']), class_mode="raw", 
                                          target_size=(img_size,img_size), color_mode='rgb', batch_size=batch_size, seed=seed)

    while True:
            
            mult_input_data = data_iter.next()
            
            return ([mult_input_data[0], mult_input_data[1][:,1:3]]*patientSex_weight, mult_input_data[1][:,0])
            
gen_flow = gen_flow_for_two_inputs(training_df)
gen_flow_train = gen_flow_for_two_inputs(validating_df)
pretrained_model = Xception(include_top=False, weights="../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",input_shape=(img_size,img_size,3))
number_of_frozen_layers = 0
for i, layer in enumerate(pretrained_model.layers):
    if i>=number_of_frozen_layers:
        break
    layer.trainable = False
img_size = (299)

input_img = Input(shape=(img_size, img_size, 3))
input_patientSex = Input(shape=(2,))

x = pretrained_model(input_img)
n = GlobalAveragePooling2D()(x)
b = concatenate([n,input_patientSex])
f = Dense(512, activation='relu',activity_regularizer=l2(0.01),kernel_regularizer=l1(0.01))(b)
f = Dense(1, activation='linear')(f)

model = Model([input_img, input_patientSex],f)
model.summary()
num_epochs = 5
learning_rate = 1e-4

weights_filepath = path+'NeuralNet_Gender/model.12-06--{epoch:02d}-{val_loss:.2f}.h5'


callbacks = [ModelCheckpoint(weights_filepath, monitor='val_loss', mode='min', verbose=2, save_best_only=True, save_freq='epoch'), 
             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=2, mode="min", min_delta=0.01, cooldown=5, min_lr=0),
             EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)]

mse = MeanSquaredError()        

model.compile(loss = mse, optimizer = Adam(lr = learning_rate), metrics = ['acc'])
history = model.fit(gen_flow[0], gen_flow[1], epochs=num_epochs, verbose=2, validation_data=(gen_flow_train[0], gen_flow_train[1]), 
                    shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, callbacks=callbacks)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('acurácia')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('época')
plt.legend(['treino', 'validação'], loc = 'upper left')
plt.show()
X = []
for i in tqdm(X_test['fileName']):
    image = load_img(path='../input/i2a2-bone-age-regression/images/'+i, grayscale=False, color_mode='rgb', target_size=(299,299),interpolation='nearest')
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch. 
    predictions = model.predict((input_arr,[X_test.loc[X_test.fileName==i][['patientSex_F','patientSex_M']]]))
    
    X.append(predictions.item())
X = []
for i in tqdm(test_df['fileName']):
    im = Image.open(os.path.join('../input/i2a2-bone-age-regression/images/'+i)).convert("RGB")
    im = im.resize((img_size,img_size))
    im = np.asarray(im)/255
    #im = np.expand_dims(im, axis=2)
    X.append(im)
X_t = np.stack(X,axis=0) 
y_t = model.predict(X_t,verbose=1)
print(y_t)
test_df['boneage'] = y_t
test_df.to_csv('\submission.csv',index=False)