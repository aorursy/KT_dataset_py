# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import cv2 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        im_path=os.path.join(dirname, filename)

        print(im_path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_excel('/kaggle/input/class-labels/y_labels.xlsx')

train_image_arr=[]

for i in range(df.shape[0]):

    image=cv2.imread('/kaggle/input/dataset/'+df.iloc[i].image_name+'.jpg')

    im_reshaped=np.reshape(image, image.shape[0]*image.shape[1]*3)

    train_image_arr.append(im_reshaped)

X_train=pd.DataFrame(train_image_arr)

y_train= df['label']



from sklearn.naive_bayes import GaussianNB

nbmodel=GaussianNB()

nbmodel.fit(X_train,y_train)


df1=pd.read_excel('/kaggle/input/test-labels1/test_y_labels.xlsx')

test_image_arr=[]

for i in range(df1.shape[0]):

    image=cv2.imread('/kaggle/input/test-dataset/'+df1.iloc[i].image_name+'.jpg')

    im_reshaped=np.reshape(image, image.shape[0]*image.shape[1]*3)

    test_image_arr.append(im_reshaped)

X_test=pd.DataFrame(test_image_arr)

y_test= df1['label']





y_predict=nbmodel.predict(X_test)



print(y_predict)



print("Score: ",nbmodel.score(X_test,y_test))
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)

print(svc.predict(X_test))

print("Score in svc: ",svc.score(X_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

print(knn.predict(X_test))



print("Score in KNN: ",knn.score(X_test,y_test))