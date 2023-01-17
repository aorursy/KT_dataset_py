from IPython.display import YouTubeVideo

YouTubeVideo('hXYd0WRhzN4', width=800, height=450)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pydicom, numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import json

from PIL import Image

from IPython.display import display

import os

import plotly.graph_objects as px

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode

from plotly.offline import iplot

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from tensorflow.keras.applications import DenseNet121

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold



import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

subs_df = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')



image_path = '../input/siim-isic-melanoma-classification/train/'



print('The size of training data : {}'.format(train_df.shape))

print('The size of testing data : {}'.format(test_df.shape))
train_df.head()
a = np.mean(train_df.target)

print('The Distribution of Training dataset : {}'.format(a))
plt.figure(figsize = (17,7))

percent_missing = train_df.isnull().sum() / (train_df.shape[0])*100

percent_missing.iplot(kind = 'bar',color ='blue')
train_df.describe()
print("The total patient ids are {}, from those the unique ids are {}".format(train_df['patient_id'].count(),train_df['patient_id'].value_counts().shape[0] ))
benign_gender = train_df.groupby(['benign_malignant']).count()['sex'].to_frame()

benign_gender.head()
#target and age

plt.figure(figsize = (17,7))

sns.boxplot(x = train_df['target'], y = train_df['age_approx'])
feature_list = ['sex','age_approx','anatom_site_general_challenge'] 

for i in feature_list: 

    train_df[i].value_counts(normalize=True).to_frame().iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='blue',

                                                      theme='pearl',

                                                      bargap=0.8,

                                                      gridcolor='white',                                                     

                                                      title=f'<b>Distribution of {i} in train set.</b>')



    test_df[i].value_counts(normalize=True).to_frame().iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='green',

                                                      theme='pearl',

                                                      bargap=0.8,

                                                      gridcolor='white',                                                     

                                                      title=f'<b>Distribution of {i} in test set.</b>')
im = train_df['image_name'].values

display(im)
plt.figure(figsize=(17,6))



image_dir = '../input/siim-isic-melanoma-classification/'



img = [np.random.choice(im + '.jpg') for i in range(10)]

img_dir = image_dir + '/jpeg/train'



for i in range(9):

    plt.subplot(3,3, i+1)

    images = plt.imread(os.path.join(img_dir, img[i]))

    plt.imshow(images)
malignant = train_df[train_df['benign_malignant']=='malignant']

benign = train_df[train_df['benign_malignant']=='benign']



print(malignant)

print(benign)
malignant.head(5)
benign.head(5)
im_malignant = malignant['image_name'].values

image_dir = '../input/siim-isic-melanoma-classification/'



img = [np.random.choice(im_malignant + '.jpg') for i in range(10)]

img_dir = image_dir + '/jpeg/train'

plt.figure(figsize=(17,17))



for i in range(9):

    plt.subplot(3,3, i+1)

    images = plt.imread(os.path.join(img_dir, img[i]))

    plt.imshow(images)

    plt.axis('off')

plt.tight_layout()

print("Random Malignant Images are Displayed!!")
im_benign = benign['image_name'].values

image_dir = '../input/siim-isic-melanoma-classification/'



img = [np.random.choice(im_benign + '.jpg') for i in range(10)]

img_dir = image_dir + '/jpeg/train'

plt.figure(figsize=(17,17))



for i in range(9):

    plt.subplot(3,3, i+1)

    images = plt.imread(os.path.join(img_dir, img[i]))

    plt.imshow(images)

    plt.axis('off')

plt.tight_layout()

print('Random Benign Images are displayed!!')
plt.imshow(pydicom.dcmread(image_path + list(train_df['image_name'])[1] + '.dcm').pixel_array)

plt.savefig('x.jpg')
plt.figure(figsize=(17,17))



for i in range(9):

    plt.subplot(3,3, i+1)

    images = pydicom.dcmread(image_path + train_df[train_df['benign_malignant']=='benign']['image_name'][i] + '.dcm')

    plt.imshow(images.pixel_array)

    plt.axis('off')

print("--- Benign DICOM Images ---")

plt.tight_layout()
plt.figure(figsize = (10,10))

data = train_df.benign_malignant.value_counts()

data.iplot(kind = 'bar', color='blue', title = 'Data Imbalance')
plt.figure(figsize = (20,15))

sns.boxplot(x = train_df['diagnosis'], y = train_df['age_approx'])
img = im_benign[0] + '.jpg'

f = plt.figure(figsize=(15,10))

f.add_subplot(1,2,1)



images = plt.imread(os.path.join(img_dir, img))

plt.imshow(images, cmap='gray')

plt.axis('off')

plt.colorbar()

plt.title('Benign Images')



f.add_subplot(1,2,2)

_= plt.hist(images[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(images[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(images[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.show()
img = im_malignant[0] + '.jpg'

f = plt.figure(figsize=(15,10))

f.add_subplot(1,2,1)



images = plt.imread(os.path.join(img_dir, img))

plt.imshow(images, cmap='gray')

plt.axis('off')

plt.colorbar()

plt.title('Malignant Images')



f.add_subplot(1,2,2)

_= plt.hist(images[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

_ = plt.hist(images[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)

_ = plt.hist(images[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

_ = plt.xlabel('Intensity Value')

_ = plt.ylabel('Count')

_ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

plt.show()
from sklearn import model_selection
#using the train.csv data

train_df['kfold'] = -1

train_df = train_df.sample(frac=1).reset_index(drop=True)

y = train_df.target.values

kf = model_selection.StratifiedKFold(n_splits=10)



for f, (t_, v_) in enumerate(kf.split(X=train_df, y=y)):

    train_df.loc[v_, 'kfold'] = f



train_df.to_csv("train_folds.csv", index=False)
def train(fold):

    train_path = "../input/siic-isic-224x224-images/train/"

    df = pd.read_csv('/kaggle/working/train_folds.csv')

    train_batch = 32

    valid_batch = 16

    epochs = 30

    

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)



    model = DenseNet121(pretrained="imagenet", include_top=False)



    train_aug = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

    

    valid_aug = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

    



    train_images = df_train.image_name.values.tolist()

    train_images = [os.path.join(train_path, i + ".png") for i in train_images]

    train_targets = df_train.target.values



    valid_images = df_valid.image_name.values.tolist()

    valid_images = [os.path.join(training_path, i + ".png") for i in valid_images]

    valid_targets = df_valid.target.values



    train_dataset = train_aug.flow_from_directory(

        image_paths=train_images,

        targets=train_targets,

        resize=None

    )

    

    



    valid_dataset = valid_aug.flow_from_directory(

        image_paths=valid_images,

        targets=valid_targets,

        resize=None

     )



    optimizer = tensorflow.keras.optimizers.Adam(model.parameters(), lr=1e-4)

    scheduler = tensorflow.keras.callbacks.ReduceLROnPlateau(

        monitor = 'val_loss',

        patience=3,

        min_lr = 0.001,

        mode="max"

    )

    es = EarlyStopping(patience=5, mode="max")



    for epoch in range(epochs):

        train_loss = Engine.train(train_dataset, model, optimizer)

        predictions, valid_loss = Engine.evaluate(

             valid_dataset, model

        )

        predictions = np.vstack((predictions)).ravel()

        auc = metrics.roc_auc_score(valid_targets, predictions)

        print(f"Epoch = {epoch}, AUC = {auc}")

        scheduler.step(auc)



        es(auc, model, model_path=f"model_fold_{fold}.bin")

        if es.early_stop:

            print("Early stopping")

            break

import xgboost as xgb
train_df['sex'] = train_df['sex'].fillna('na')

train_df['age_approx'] = train_df['age_approx'].fillna(0)

train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].fillna('na')



test_df['sex'] = test_df['sex'].fillna('na')

test_df['age_approx'] = test_df['age_approx'].fillna(0)

test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].fillna('na')
train_df['sex'] = train_df['sex'].astype("category").cat.codes +1

train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].astype("category").cat.codes +1

train_df.head()
test_df['sex'] = test_df['sex'].astype("category").cat.codes +1

test_df['anatom_site_general_challenge'] = test_df['anatom_site_general_challenge'].astype("category").cat.codes +1

test_df.head()
x_train = train_df[['sex', 'age_approx','anatom_site_general_challenge']]

y_train = train_df['target']





x_test = test_df[['sex', 'age_approx','anatom_site_general_challenge']]

#y_test = test_df['target']





train_DMatrix = xgb.DMatrix(x_train, label= y_train)

test_DMatrix = xgb.DMatrix(x_test)
model = xgb.XGBClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

y_pred
subs_df.target = model.predict_proba(x_test)[:,1]

sub_tabular = subs_df.copy()
subs_df.to_csv('submissions.csv', index = False)

print('Successfull!!')
subs_df.head()