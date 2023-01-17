! pip install imutils

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from imutils import paths

import numpy as np

import imutils 

import cv2 

import os


dataset_train = "../input/2019-fall-pr-project/train/train"



X = []

Y = []



for i, img in enumerate(paths.list_images(dataset_train)):

  

  # dog: 1

  Y.append(1 if 'dog' in img else 0)  

  img = cv2.imread(img)

#  print(img.shape)

  img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

#  print(img.shape)

  img = img.flatten()

#  print(img.shape)

  X.append(img)

  

  print("loading... ", str(i/200)+"%") if i%2000==0 else None

  

  if i==1000:

    break



X = np.array(X)

Y = np.array(Y)

print(X.shape, Y.shape)
X_train, X_val, Y_train, Y_val = train_test_split(

    X, Y, test_size=0.25, random_state=42)
%matplotlib inline

#import numpy as np

import matplotlib.pyplot as plt

#from scipy import stats



# use seaborn plotting defaults

#import seaborn as sns; sns.set()



#import time



from sklearn.svm import SVC

#from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
svc = SVC(kernel='poly')

svc.fit(X_train, Y_train)
predict = svc.predict(X_val)

print(classification_report(Y_val, predict))


dataset_test = "../input/2019-fall-pr-project/test1/test1"



X_test = []

#Y_test = []



ids = []



for i, img in enumerate(paths.list_images(dataset_test)):

  

  ids.append(os.path.basename(img)[:-4])

  

  # dog: 1

  #Y_test.append(1 if 'dog' in img else 0)  

  img = cv2.imread(img)

#  print(img.shape)

  img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

#  print(img.shape)

  img = img.flatten()

#  print(img.shape)

  X_test.append(img)

  

  print("loading... ", str(i/50+10)+"%") if i%500==0 else None

  

ids = np.array(ids)

X_test = np.array(X_test)

#Y_test = np.array(Y_test)

print(ids.shape)

print(X_test.shape)
result = svc.predict(X_test)


# numpy 를 Pandas 이용하여 결과 파일로 저장





import pandas as pd



print(result.shape)

df = pd.DataFrame(result, columns=['label'])

#df = df.replace('dog',1)

#df = df.replace('cat',0)

df.index += 1

df.index.name = 'id'

df.to_csv('result_namhun_kim-2320.csv',index=True, header=True)