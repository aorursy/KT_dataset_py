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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv') #adding data

#print 1st 5 rows

dataset.head()
dataset.isnull().sum()
dataset.dtypes
data = dataset.drop(['education','BPMeds','totChol','BMI','glucose','heartRate' ], axis = 'columns')

data.head()
cigarettes = data['cigsPerDay']

cigarettes.head()

#cig = cigs.mean()
cig = cigarettes.mean()
import math

integer_value = math.floor(cig)

integer_value
cigarettes.fillna(integer_value, inplace = True)

#hr.fillna(integer_value, inplace = True)

data.isnull().sum()
#this data is labeled data

#independent variable (X) will be Age, Estimated Salary

#dependent variable(y) will be Purcahsed (It will be lable)



#define X,y

X = dataset.iloc[:, [ 1, 3]].values

y = dataset.iloc[:, 15].values



#print X,y

print(X[:5, :])

print(y[:5])
#import train_test_split from sklearn to train and test our data

from sklearn.model_selection import train_test_split 

#define 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



#print

print(X_train[:5],'\n', '\n', y_train[:5],'\n','\n', X_test[:5],'\n','\n', y_test[:5])
#dataset.info()

#check corelation among attributes of dataset 

dataset.corr()
#normalization and feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
print(X_train[:5], '\n','\n', X_test[:5])
#create object of LogisticRegression class to refer as classifier

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print(X_test[:5], '\n', '\n', y_pred[:5])
print(y_pred[:12],'\n','\n',y_test[:12])
#probability is 11/12 i.e 91.66% so, good to go



#we'll use a Confusion Matrix to evaluate exactly how accurate our Logistic Regression model is,

#the more the matching values, more is the accuracy



from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)

print(matrix)
X = data[['male','age','currentSmoker','cigsPerDay','prevalentStroke','prevalentHyp','diabetes','sysBP','diaBP']]

y =  data['TenYearCHD']

X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 99)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test,y_test)