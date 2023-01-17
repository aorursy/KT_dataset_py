# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2
cats_images = []

path = "../input/dog vs cat/dataset/training_set/cats"

for image in os.listdir(path):

    full_path = os.path.join(path,image)

    img = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img,(128,128))

    cats_images.append(img)
cat = np.concatenate(cats_images)

cat.shape
df1 = pd.DataFrame(cat)
df1['label']=0
df1.head
dogs_images = []

path = "../input/dog vs cat/dataset/training_set/dogs"

for image in os.listdir(path):

    full_path = os.path.join(path,image)

    img = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img,(128,128))

    dogs_images.append(img)

dog=np.concatenate(dogs_images)

df2 = pd.DataFrame(dog)

df2['label']=1
df2.head()
df = pd.concat([df1,df2])
df.head()
df.shape
features = df.drop("label",axis=1)

target = df['label']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features,target)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
from PIL import Image
Image.open("../input/dog vs cat/dataset/training_set/cats/cat.1.jpg")
path = "../input/dog vs cat/dataset/training_set/cats/cat.1.jpg"

img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img,(128,128))
img.shape
prediction = model.predict(img)
prediction



np.mean(prediction == 1)
np.mean(prediction == 0)