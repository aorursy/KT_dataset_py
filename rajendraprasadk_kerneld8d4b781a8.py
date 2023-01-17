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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
fr=data["Embarked"].mode()
fr[0]
data["Embarked"].fillna(fr[0], inplace=True)
data["Age"].fillna(30, inplace=True)
from sklearn import preprocessing
label_encoder_sex = preprocessing.LabelEncoder()
data['Sex']=label_encoder_sex.fit_transform(data['Sex']) 

label_encoder_embarked = preprocessing.LabelEncoder()
data['Embarked']=label_encoder_embarked.fit_transform(data['Embarked'])
data_train_X=data[['Pclass','Sex', 'Age', 'SibSp',
       'Parch','Fare','Embarked']]
data_train_y=data['Survived']
# Test data preprocessing based on training data
data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
data_test["Age"].fillna(30, inplace=True)
data_test["Fare"].fillna(32, inplace=True)
data_test['Embarked']=label_encoder_embarked.transform(data_test['Embarked']) 
data_test['Sex']=label_encoder_sex.transform(data_test['Sex'])
data_test_X=data_test[['Pclass','Sex', 'Age', 'SibSp',
       'Parch','Fare','Embarked']]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
clf_lr=LogisticRegression()
clf_lr.fit(data_train_X,data_train_y)
y_pre=clf_lr.predict(data_test_X)
y_pre
data_test_y=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data_test_y=data_test_y['Survived']
results = confusion_matrix(data_test_y,y_pre) 
print('Logistic Regression model metrics:') 
print('----------------------------------------------------')
print ('Confusion Matrix :')
print('----------------------------------------------------')
print(results) 
print('----------------------------------------------------')
print ('Accuracy Score :',accuracy_score(data_test_y,y_pre))
print('----------------------------------------------------')
print ('Report : ')
print (classification_report(data_test_y,y_pre))