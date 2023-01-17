# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile



with zipfile.ZipFile("../input/leaf-classification/images.zip", 'r') as zipf:

    zipf.extractall(".")

with zipfile.ZipFile("../input/leaf-classification/train.csv.zip", 'r') as zipf:

    zipf.extractall(".")

with zipfile.ZipFile("../input/leaf-classification/test.csv.zip", 'r') as zipf:

    zipf.extractall(".")    
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import matplotlib.image as image
train = pd.read_csv("./train.csv")

train.head(10)

test = pd.read_csv("./test.csv")

test.head(10)



print(train.shape, test.shape)
images = []

for im in os.listdir("./images/"):

    images.append(im)



for im in images[45:50]:

    img = image.imread("./images/"+im)

    plt.figure()

    plt.imshow(img, cmap='binary')
X = train.drop(['id', 'species'], axis=1)

y = train['species']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, shuffle=True, stratify=y)



labenc = LabelEncoder()

y_train = labenc.fit_transform(y_train)

y_val = labenc.transform(y_val)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)
# Working very nice & getting good log_loss score of 0.077 on validation data & 0.111 on unseen test data. 



logreg = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred_log = logreg.predict_proba(X_val)



print('train score : ', logreg.score(X_train, y_train))

print('val score : ', logreg.score(X_val, y_val))

print('log loss : ', log_loss(y_val, y_pred_log))
# Not working good on testing data instead of getting good score on validation & training set.

# Getting log_loss score of 2.313 on validation data & 2.308 on unseen test data.



svmclf = SVC(probability=True).fit(X_train, y_train)

y_pred_svm = svmclf.predict_proba(X_val)



print('train score : ', svmclf.score(X_train, y_train))

print('val score : ', svmclf.score(X_val, y_val))

print('log loss : ', log_loss(y_val, y_pred_svm))
# It is also not working good enough instead of getting 100% accuracy on training set & 97.98% accuracy \

# on validation set. Getting log_loss score of 0.703 on validation set & 0.744 on unseen test set.



rfclf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

y_pred_rf = rfclf.predict_proba(X_val)



print("train score : ", rfclf.score(X_train, y_train))

print("val score : ", rfclf.score(X_val, y_val))

print("log loss : ", log_loss(y_val, y_pred_rf))
X_test = test.drop('id', axis=1)

X_test = scaler.transform(X_test)

y_test = logreg.predict_proba(X_test)



#y_test_rf = rfclf.predict_proba(X_test)

#y_test_svm = svmclf.predict_proba(X_test)
# Total number of species is 99 & are using as column names in the submission file.

cols = sorted(train.species.unique())



res = pd.DataFrame(data = y_test, columns = cols)

result = pd.DataFrame(data = test['id'], columns=['id'])

        

result = pd.concat([result, res], axis=1)

result.head()



# Writing/Converting dataframe into .csv files.

result.to_csv("leaf_classification.csv", header=True, index=False)

result.head(10)