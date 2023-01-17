# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import warnings



import matplotlib.pyplot as plt

%matplotlib inline



warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/UnivBank.csv")
df.head()
df.tail()
len(df)
df.shape
df.columns
df['Personal Loan'].value_counts()
df['Income'].unique()

# df['Income'].plot(kind='box')
# df['ZIP Code'].unique()
df['Family'].unique()
df['CCAvg'].unique()

# df['CCAvg'].plot(kind='box')
df['Education'].value_counts()
# df['Mortgage'].unique()
df['Securities Account'].value_counts()
df['CD Account'].value_counts()
df['Online'].value_counts()
df['CreditCard'].value_counts()
df.dtypes
df.duplicated().sum()
df.isna().sum()
df['Experience'].unique()
df['Experience'].replace(-1, 0, inplace=True)

df['Experience'].replace(-2, 0, inplace=True)

df['Experience'].replace(-3, 0, inplace=True)
df['Experience'].unique()
df['Age'].unique()
x = df.drop(columns=['Personal Loan'])

x.columns
y = df[['Personal Loan']]

y.columns
def split_fit_predict(algorithm, x, y, scaler = None, test_size = 0.25, kernel = 'rbf', C = 1.0, degree = 3):

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size)

    if(algorithm == 'Logistic'):

        model = LogisticRegression()

        model.fit(train_x, train_y)

        predict_y = model.predict(test_x)

        print("Accuracy Score: ", accuracy_score(test_y, predict_y))

        predict_y

    elif(algorithm == 'SVC'):

        train_x_scaled = scaler.fit_transform(train_x)

        test_x_scaled = scaler.transform(test_x)

        model = SVC()

        model.fit(train_x_scaled, train_y)

        predict_y = model.predict(test_x_scaled)

        print("Accuracy Score: ", accuracy_score(test_y, predict_y))

        print("\n")

        predict_y

        
print("Logistic Regression - Prediction:")

split_fit_predict('Logistic', x, y, 0.30)
print("SVM - SVC - Prediction:")

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

    print("For kernel '{0}': ".format(kernel))

    split_fit_predict('SVC', x, y, StandardScaler(), 0.30, kernel)
for c in np.logspace(3,-1, base = 2, num = 6):

    print("For penalty '{0}':".format(c))

    split_fit_predict('SVC', x, y, StandardScaler(), 0.30, 'sigmoid', c)
for degree in range(2,6):

    print("For degree '{0}':".format(degree))

    split_fit_predict('SVC', x, y, StandardScaler(), 0.30, 'sigmoid', 2.63, degree)
original_score = 0.9144

best_score = 0.9746666666666667

improvement = np.abs(np.round(100*(original_score - best_score)/original_score,2))

print('overall improvement is {} %'.format(improvement))