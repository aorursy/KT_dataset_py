from IPython.display import Image
Image("../input/assets/assets/RetinopatiaDiabetica.jpg")
Image('../input/assets/assets/tensorflow.PNG')
Image('../input/assets/assets/diagnosisHist.png')
Image('../input/assets/assets/props.PNG')
import pandas as pd

import matplotlib.pyplot as plt

import os # Para importar las imagenes

from PIL import Image # Para visualizar las imagenes

import numpy as np
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
train_df['path'] = train_df['id_code'].map(lambda x: os.path.join('../input/aptos2019-blindness-detection/train_images/','{}.png'.format(x)))

train_df['path_gray'] = train_df['id_code'].map(lambda x: os.path.join('../input/transfrom-images/transform_images/','{}.png'.format(x)))
no_dr = Image.open(train_df[train_df.diagnosis==0].iloc[0].path)

plt.imshow(np.asarray(no_dr))
mild = Image.open(train_df[train_df.diagnosis==1].iloc[0].path)

plt.imshow(np.asarray(mild))
moderate = Image.open(train_df[train_df.diagnosis==2].iloc[0].path)

plt.imshow(np.asarray(moderate))
sev = Image.open(train_df[train_df.diagnosis==3].iloc[0].path)

plt.imshow(np.asarray(sev))
prol = Image.open(train_df[train_df.diagnosis==4].iloc[0].path)

plt.imshow(np.asarray(prol))
train_df = train_df[:610]
diagnosis_labs = pd.read_csv('../input/diagnosis/diagnosis.csv')

train_df_labs = pd.merge(train_df, diagnosis_labs)

train_df_labs["diagnosis_n"].value_counts().plot(kind='bar')
# Ver dimensiones de imagen

im = Image.open(train_df['path'][1])

width, height = im.size

print(width,height) 

plt.imshow(np.asarray(im))
train, test = np.split(train_df.sample(frac=1), [int(.65*len(train_df))])
train_labs = pd.merge(train, diagnosis_labs)

train_labs["diagnosis_n"].value_counts().plot(kind='bar')
max_size = train_labs['diagnosis_n'].value_counts().max()

min_size = train_labs['diagnosis_n'].value_counts().min()



lst = [train_labs]

for class_index, group in train_labs.groupby('diagnosis'):

   lst.append(group.sample(max_size-len(group), replace=True))

oversampled_train = pd.concat(lst)
oversampled_train["diagnosis_n"].value_counts().plot(kind='bar')
no_dr = np.random.choice(train_labs[train_labs.diagnosis==0].index, min_size, replace=False)

mild = np.random.choice(train_labs[train_labs.diagnosis==1].index, min_size, replace=False)

moderate = np.random.choice(train_labs[train_labs.diagnosis==2].index, min_size, replace=False)

severe = np.random.choice(train_labs[train_labs.diagnosis==3].index, min_size, replace=False)

proliferative = np.random.choice(train_labs[train_labs.diagnosis==4].index, min_size, replace=False)
undersample_indexes = np.concatenate([no_dr,mild,moderate,severe,proliferative])
undersampled_train = train_labs.loc[undersample_indexes]

undersampled_train["diagnosis_n"].value_counts().plot(kind='bar')
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm
test_images = []

for i in tqdm(range(test.shape[0])):

    img = image.load_img(test['path_gray'].iloc[i], target_size=(256,256, 1), color_mode="grayscale")

    img = image.img_to_array(img)

    img = img/255

    test_images.append(img)

X_test = np.array(test_images)



y_test = test['diagnosis'].values

y_test = to_categorical(y_test)