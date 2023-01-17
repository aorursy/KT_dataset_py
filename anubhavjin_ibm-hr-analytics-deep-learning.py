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
data=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.shape
data.describe()
data.columns
dataset = data[['Attrition',
                   'Age',
                   'BusinessTravel',
                   'DailyRate',
                   'Department',
                   'DistanceFromHome',
                   'Education',
                   'EducationField',
                   'EmployeeCount',
                   'EnvironmentSatisfaction',
                   'Gender',
                   'HourlyRate',
                   'JobInvolvement',
                   'JobLevel',
                   'JobRole',
                   'JobSatisfaction',
                   'MaritalStatus',
                   'MonthlyIncome',
                   'MonthlyRate',
                   'NumCompaniesWorked',
                   'OverTime',
                   'PercentSalaryHike',
                   'PerformanceRating',
                   'RelationshipSatisfaction',
                   'StockOptionLevel',
                   'TotalWorkingYears',
                   'TrainingTimesLastYear',
                   'WorkLifeBalance',
                   'YearsAtCompany','YearsInCurrentRole',
                   'YearsSinceLastPromotion',
                   'YearsWithCurrManager']]

dataset.head()
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_6= LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_9= LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_13= LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])
labelencoder_X_15= LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
labelencoder_X_19= LabelEncoder()
X[:, 19] = labelencoder_X_19.fit_transform(X[:, 19])
X = X.astype(float)
labelencoder_y= LabelEncoder()
y = labelencoder_y.fit_transform(y)
d=pd.DataFrame(X)
d.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dropout = 0.1
epochs = 100
batch_size = 30
optimizer = 'adam'
k = 20

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(16, kernel_initializer="truncated_normal", activation = 'relu', input_shape = (X.shape[1],)))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(1, kernel_initializer="truncated_normal", activation = 'sigmoid', )) #outputlayer
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier ,batch_size=batch_size,epochs=50,verbose=1)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 3)
accuracies
accuracies.mean()
accuracies.max()

