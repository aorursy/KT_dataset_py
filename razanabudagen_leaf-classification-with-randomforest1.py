import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2 as cv

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import load_img

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import  train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import zipfile



with zipfile.ZipFile('/kaggle/input/leaf-classification/sample_submission.csv.zip') as z_samp:

    z_samp.extractall()

    

with zipfile.ZipFile('/kaggle/input/leaf-classification/train.csv.zip') as z_train:

    z_train.extractall()



with zipfile.ZipFile('/kaggle/input/leaf-classification/images.zip') as z_img:

    z_img.extractall()

    

with zipfile.ZipFile('/kaggle/input/leaf-classification/test.csv.zip') as z_test:

    z_test.extractall()
os.listdir()
len(os.listdir('images'))
plt.figure(figsize=(20,15))



for i in range(25):

    j=np.random.choice((os.listdir('images')))

    plt.subplot(5,5,i+1)

    img=load_img(os.path.join('/kaggle/working/images',j))

    plt.imshow(img)
t_data=pd.read_csv('train.csv',index_col=False)

test_data=pd.read_csv('test.csv',index_col=False)
t_data.head()
t_data.isnull().any().sum()
test_data.isnull().any().sum()
t_data.info()
t_data['species'].nunique()
x = t_data.drop('species',axis=1)

y = t_data['species']
encoder = LabelEncoder()



y_fit = encoder.fit(t_data['species'])

y_label = y_fit.transform(t_data['species']) 

classes = list(y_fit.classes_) 



classes
len(classes)
t_data=t_data.drop(['id','species'],axis=1)



test_id=test_data.id

#test_data=test_data.drop(['id'],axis=1)
t_data
# splitting



x_train, x_test, y_train, y_test = train_test_split(x,y_label, test_size = 0.2, random_state =1)
#Random Forest Classifier



classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

print (classification_report(y_test, predictions))
final_predictions = classifier.predict_proba(test_data)
submission = pd.DataFrame(final_predictions, columns=classes)

submission.insert(0, 'id', test_id)



submission.reset_index()
submission.to_csv('submission.csv', index = False)