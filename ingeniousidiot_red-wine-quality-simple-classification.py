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





from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

# Any results you write to the current directory are saved as output.
#loading dataset and expolring how the data is distributed

data=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
#description of the dataset

data.describe()
#checking for any null values in the dataset

data.isnull().sum()
cols=['fixed acidity','volatile acidity','citric acid','residual sugar',

      'chlorides','free sulfur dioxide','total sulfur dioxide','sulphates','alcohol']

for i in cols:

    sns.barplot(x = 'quality', y = i, data = data)

    plt.show()
#Making binary classificaion for the response variable.

#Dividing wine as low and hingh by giving the limit for the quality

bins = (2, 6.5, 8)

group_names = ['low', 'high']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
#Handling categorical feature quality

from sklearn.preprocessing import LabelEncoder

label_quality = LabelEncoder()

data['quality'] = label_quality.fit_transform(data['quality'])
#correlation between features

data.corr()

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot=True)

plt.show()
#Now seperating the dataset as label variable and feature variabe

x=data.drop('quality',axis=1)

y=data['quality']
#spliting the dataset as train set and test set

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#Since the range of feature variables differ significantly so scaling them using Standard Scaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
#Making the model 1 by using Random Forest Classifier

model1 = RandomForestClassifier(n_estimators=100)

model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)
#checking for confusion matrix and clssification report

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
#Making the model 2 by using SGD Classifier

model2 = SGDClassifier(penalty=None)

model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
#Making the model 2 by using SGD Classifier

model3 = SVC()

model3.fit(x_train, y_train)

y_pred3 = model3.predict(x_test)
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))