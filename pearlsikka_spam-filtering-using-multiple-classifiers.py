# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

df.shape
df.head()
df= df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

df= df.rename(columns={'v2':'text','v1':'label'})
df.head()
df.isnull().any()        #check if there is any null data
from sklearn.feature_extraction.text import CountVectorizer



cv=CountVectorizer(stop_words='english')



feature_vectors= cv.fit_transform(df['text'])

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(feature_vectors,df['label'],test_size=0.3, random_state=42)

print(np.shape(x_train), np.shape(x_test),np.shape(y_train))

print('There are {} samples in the training set and {} samples in the test set'.format(

x_train.shape[0], x_test.shape[0]))

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



model_SVM = SVC()

model_SVM.fit(x_train, y_train)



y_pred_SVM = model_SVM.predict(x_test)



print("Training Accuracy using SVM :", model_SVM.score(x_train, y_train))

print("Testing Accuracy using SVM:", model_SVM.score(x_test, y_test))



print('Confusion matrix')

cm1 = confusion_matrix(y_test, y_pred_SVM)

print(cm1)



print("Accuracy Score for Test Set using SVM:", accuracy_score(y_test, y_pred_SVM))
from sklearn.naive_bayes import MultinomialNB



model_NB= MultinomialNB()

model_NB.fit(x_train,y_train)



y_pred_NB=model_NB.predict(x_test)



print("Training Accuracy using Naive Bayes:", model_NB.score(x_train, y_train))

print("Testing Accuracy using Naive Bayes:", model_NB.score(x_test, y_test))



print('Confusion matrix')

cm2 = confusion_matrix(y_test, y_pred_NB)

print(cm2)



from sklearn.metrics import accuracy_score

print("Accuracy Score for Test Set using Naive Bayes:", accuracy_score(y_test, y_pred_NB))
from sklearn.ensemble import RandomForestClassifier



model_RF=RandomForestClassifier(n_estimators=31, random_state=111)

model_RF.fit(x_train,y_train)



y_pred_RF=model_RF.predict(x_test)



print("Training Accuracy using Random Forest:", model_RF.score(x_train, y_train))

print("Testing Accuracy using Random Forest:", model_RF.score(x_test, y_test))



print('Confusion matrix')

cm3 = confusion_matrix(y_test, y_pred_RF)

print(cm3)



from sklearn.metrics import accuracy_score

print("Accuracy Score for Test Set using Random Forest:", accuracy_score(y_test, y_pred_RF))