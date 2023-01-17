!pip install PyDrive

import os

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials
auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools

!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null

!apt-get update -qq 2>&1 > /dev/null

!apt-get -y install -qq google-drive-ocamlfuse fuse

from google.colab import auth

auth.authenticate_user()

from oauth2client.client import GoogleCredentials

creds = GoogleCredentials.get_application_default()

import getpass

!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL

vcode = getpass.getpass()

!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
!mkdir -p drive

!google-drive-ocamlfuse drive
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm

from PIL import Image

from sklearn.model_selection import train_test_split
download = drive.CreateFile({'id': '18s0FOQ-eR-RVuyrpS0sjzh9Lud8TXShg'})

download.GetContentFile('HAM10000_metadata.csv')
image_file = pd.read_csv('/content/drive/HAM10000_images/HAM10000_metadata.csv')
image_file.head()
lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

image_file['le_type'] = image_file['dx'].map(lesion_type_dict.get) 

image_file['le_type_idx'] = pd.Categorical(image_file['le_type']).codes

image_file.isnull().sum()
image_file['age'].fillna((image_file['age'].median()), inplace=True)



image_file['le_type'].value_counts().plot(kind='bar')
image_file['localization'].value_counts().plot(kind='bar')
image_file['sex'].value_counts().plot(kind='bar')
image_file['age'].value_counts().plot(kind='bar')
image_file.dtypes
mod_image = []

for i in tqdm(range(image_file.shape[0])):

    img = image.load_img('/content/drive/HAM10000_images/'+image_file['image_id'][i]+'.jpg', target_size=(50,50,3))

    img = image.img_to_array(img)

    img = img/255

    mod_image.append(img)

    X = np.array(mod_image)
y=train['label'].values

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size=0.2)
y=image_file['le_type_idx'].values

y = to_categorical(y)
X_train, X_validate, y_train, y_validate = train_test_split(X,y, test_size = 0.1, random_state = 2)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(50,50,3)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
prediction = model.predict_classes(X_test)
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(X_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
