ls -al
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import cv2

import os



import os 

from glob import glob

from PIL import Image



from sklearn import svm, datasets

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from tqdm import tqdm_notebook
scaler = StandardScaler()

dataset_train = "../input/2019-fall-pr-project/train/train/"

train_data_list = glob(dataset_train+'*.*.jpg')

train = list()

label = list()
for i in tqdm_notebook(range(len(train_data_list))):

    try:

        if train_data_list[i][42] == 'c' :

            label.append(0)

        else :

            label.append(1)



        image = Image.open(train_data_list[i])



        image = image.resize((32,32), Image.ANTIALIAS)

        image = np.array(image)/255



        image = image.reshape(-1)



        train.append(image)

    except :

        import pdb;pdb.set_trace()
train = np.array(train)

scaler.fit(train)

train = scaler.transform(train)

label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.1, random_state=42)
y_train
parameters = {'kernel':['rbf'], 'C': [1, 5, 10, 100, 1000]}

svc = svm.SVC(gamma="scale",decision_function_shape="ovo")

clf = GridSearchCV(svc, parameters, cv=2)

clf.fit(X_train, y_train)
clf.best_params_
y_predict = clf.predict(X_test)

confusion_matrix(y_test, y_predict)
target_names = ['dog', 'cat']

print(classification_report(y_test, y_predict, target_names=target_names))
test = list()



dataset_test = "../input/2019-fall-pr-project/test1/test1/"

test_data_list = glob(dataset_test+'*.jpg')



for i in tqdm_notebook(range(len(test_data_list))):



  image = Image.open(test_data_list[i])

  image = image.resize((32,32), Image.ANTIALIAS)

  image = np.array(image)/255

  image = image.reshape(-1)



  test.append(image)



test = np.array(test)

scaler.fit(test)

test = scaler.transform(test)
result = clf.predict(test)

test = result.reshape(-1,1)
# numpy 를 Pandas 이용하여 결과 파일로 저장



import pandas as pd



print(test.shape)

df = pd.DataFrame(test, columns=["label"])

df.index = np.arange(1,len(df)+1)

df.index.name = 'id'



df.to_csv('results-jwkim-v2.csv',index=True, header=True)