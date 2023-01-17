# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = '/kaggle/input/digit-recognizer/train.csv'

test_data = '/kaggle/input/digit-recognizer/test.csv'

sample_submission = '/kaggle/input/digit-recognizer/sample_submission.csv'
df_train = pd.read_csv(train_data)

df_test = pd.read_csv(test_data)

df_sample_submission = pd.read_csv(sample_submission)
df_train.head()
df_test.head()
df_sample_submission.head()
def get_image_from_dataframe(df):

    if 'label' in df.columns:

        labels = df['label'] #getting labels from dataframe

        label_array = np.array(labels) # getting labels from array

        image_pixels = df[df.columns[1:785]]



    else:

        label_array = None

        image_pixels = df[df.columns[0:784]]

    temp_images = []

    image_array = np.array(image_pixels)

    for index in range(len(image_array)):

        new_array = np.array_split(image_array[index],28)

        new_array = np.expand_dims(new_array,axis=2)

        temp_images.append(new_array)

    image_data = np.array(temp_images).astype('float')

    return image_data,label_array
    
training_images,training_labels = get_image_from_dataframe(df_train)

testing_images , _ = get_image_from_dataframe(df_test)
print("Number of training images : ", len(training_images))

print("Number of testing images : ",len(testing_images))
print("Training data shape : " , training_images.shape)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=512,activation='relu'),

    tf.keras.layers.Dense(units=10,activation = 'softmax')

])
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['acc'])
#Normalizing Images

training_images = training_images/255

testing_images = testing_images/255
model.fit(

training_images,

training_labels,

epochs = 20

)
classifications = model.predict(testing_images)
classifications
image_id = []

predictions = []

for image_index,prediction in enumerate(classifications):

    image_id.append(image_index)

    predictions.append(np.argmax(prediction))
image_id[5]
predictions[0]
df_final_predictions = pd.read_csv(sample_submission)
df_final_predictions['Label'] = predictions
df_final_predictions.head()
df_final_predictions.to_csv('final_sumbission.csv',index=False)
# df_final_predictions.reset_index(drop=True,inplace=True)
df_final_predictions.head()
temp_test_images = []

image_pixels = df_test[df_test.columns[0:784]]

image_array = np.array(image_pixels)

for index in range(len(image_array)):

    new_array = np.array_split(image_array[index],28)

    temp_test_images.append(new_array)

test_image_data = np.array(temp_test_images).astype('float')



plt.imshow(test_image_data[2])
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe



# create a link to download the dataframe

create_download_link(df_final_predictions)
