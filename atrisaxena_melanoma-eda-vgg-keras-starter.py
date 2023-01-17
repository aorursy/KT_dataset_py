import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split

from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
# List files available
print(os.listdir("../input/siim-isic-melanoma-classification"))
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
train.columns
test.columns
missing_col = ['sex','age_approx','anatom_site_general_challenge']

fig, axes = plt.subplots(ncols = 2, figsize = (20,4),dpi = 100)
sns.barplot(x= train[missing_col].isnull().sum().index , y= train[missing_col].isnull().sum().values, ax=axes[0])
sns.barplot(x= test[missing_col].isnull().sum().index , y= test[missing_col].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size = 15, labelpad =20)

axes[0].tick_params(axis ='x', labelsize = 15)
axes[0].tick_params(axis='y', labelsize = 15)

axes[1].tick_params(axis ='x', labelsize = 15)
axes[1].tick_params(axis='y', labelsize = 15)
axes[0].set_title('Training set', fontsize = 12)
axes[1].set_title('Test set', fontsize  =12)
plt.show()
train['target'].value_counts()
fig,axes = plt.subplots(ncols =1, figsize = (6,3), dpi = 100)

sns.countplot(x = 'target', hue = 'target' , data=train)

plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
axes.set_xticklabels(['Benign(32542)', 'Melignant (584)'])

plt.title('Number of examples')
plt.show()
data = train.groupby(['target','sex'])['benign_malignant'].count().to_frame().reset_index()
ax = sns.catplot(x='target',y= 'benign_malignant', hue='sex',data=data ,kind='bar')
plt.xlabel("0: Benign, 1: Melignant")
plt.ylabel("Count of cases")
data = train.groupby(['sex','anatom_site_general_challenge'])['target'].count().to_frame().reset_index()
ax = sns.catplot(x='anatom_site_general_challenge',y= 'target', hue='sex',data=data ,kind='bar')
plt.gcf().set_size_inches(10,4)
plt.xlabel("Location of Image")
plt.ylabel("Count of Cases")
CATEGORIES = ['benign','malignant']
NUM_CATEGORIES = len(CATEGORIES)
SEED = 1987
data_dir = '../input/siim-isic-melanoma-classification/jpeg/'
train_dir = data_dir+ 'train/'
test_dir = data_dir +'test/'
sample_submission = pd.read_csv(os.path.join('../input/siim-isic-melanoma-classification', 'sample_submission.csv'))
fig = plt.figure(1, figsize=(15, 10))
grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES, 5), axes_pad=0.05)
i = 0
for category_id, category in enumerate(CATEGORIES):
    for filepath in train[train['benign_malignant'] == category]['image_name'].values[:5]:
        ax = grid[i]
        img = Image.open("../input/siim-isic-melanoma-classification/jpeg/train/"+filepath+".jpg")
        img = img.resize((240,240))
        ax.imshow(img)
        ax.axis('off')
        if i % 5 == 5 - 1:
            ax.text(250, 112, category, verticalalignment='center')
        i += 1
plt.show();
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (300, 300, 3))
for layer in model.layers[:3]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x) 
#Thanks to https://www.kaggle.com/ibtesama/siim-baseline-keras-vgg16
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float64)
        y_pred = tf.dtypes.cast(y_pred, tf.float64)
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
opt = Adam(lr=1e-4)
model_final = Model(inputs = model.input, outputs = predictions)
model_final.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)
#model_final.compile(loss = focal_loss(), optimizer = optimizers.SGD(lr=0.00001, momentum=0.9), metrics=["accuracy"])
model_final.summary()
model_final.load_weights('../input/melanoma-eda-vgg-keras-starter/vgg16_1.h5')
labels=[]
data=[]
for i in range(train.shape[0]):
    data.append(train_dir + train['image_name'].iloc[i]+'.jpg')
    labels.append(train['target'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['target']=labels
test_data=[]
for i in range(test.shape[0]):
    test_data.append(test_dir + test['image_name'].iloc[i]+'.jpg')
df_test=pd.DataFrame(test_data)
df_test.columns=['images']

X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)
train=pd.DataFrame(X_train)
train.columns=['images']
train['target']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['target']=y_val
datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=360.,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='target',
    target_size=(300, 300),
    batch_size=64,
    shuffle=True,
    class_mode='raw')

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='target',
    target_size=(300, 300),
    shuffle=False,
    batch_size=64,
    class_mode='raw')
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')
nb_epochs = 2
batch_size=64
nb_train_steps = train.shape[0]//batch_size
nb_val_steps=validation.shape[0]//batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
model_final.fit_generator(
    train_generator,
    epochs=nb_epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early])
target=[]
for path in df_test['images']:
    img=cv2.imread(str(path))
    img = cv2.resize(img, (300,300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img=np.reshape(img,(1,300,300,3))
    prediction=model_final.predict(img)
    target.append(prediction[0][0])


submission['target']=target
submission.to_csv('submission.csv', index=False)
submission.head()