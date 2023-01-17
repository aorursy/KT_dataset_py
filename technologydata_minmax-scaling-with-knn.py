# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Importing training data set
X_train=pd.read_csv("../input"+'/X_train.csv')
Y_train=pd.read_csv("../input"+"/Y_train.csv")


# Importing testing data set
X_test=pd.read_csv("../input"+"/X_test.csv")
Y_test=pd.read_csv("../input"+"/Y_test.csv")
#Print dataset to check if it loads correctly

print (X_train.head())
#Plot histogram of continous variable

X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])

# We infer that ApplicantIncome and CoapplicantIncome are in similar range (0-50000$) where as LoanAmount is in thousands and it ranges from 0 to 600$. 
# The story for Loan_Amount_Term is completely different from other variables because its unit is months as opposed to other variables where the unit is dollars.

#If we try to apply distance based methods such as kNN on these features, feature with the largest range will dominate the outcome results and we’ll obtain less accurate predictions.
#We can overcome this trouble using feature scaling. Let’s do it practically.

knn=KNeighborsClassifier(n_neighbors=5)
X=X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History']]
y=Y_train.values
knn.fit(X,y.ravel())


accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome', 'CoapplicantIncome',
                             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]))
#Currently the accuracy we got is 61 percent.Lets try to do min max scaling and then run the KNN
min_max=MinMaxScaler()
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train.values.ravel())
# Checking the model's accuracy
accuracy_score(Y_test,knn.predict(X_test_minmax))
#We got 75 percent of accuracy when we scaled the data.