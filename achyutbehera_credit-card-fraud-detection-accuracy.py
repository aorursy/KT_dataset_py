# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/creditcard.csv')
#Checking the data if any columns contains null. 
dataset.isnull().sum()
dataset['V1'].count()
#The calss is dependent varibale which specifies whether the transaction is fraud or not.
#Splitting the data into X and y
X = dataset.iloc[:, 0:31].values
Y = dataset.iloc[:,-1].values


print(len(X[0]))
#Spliting the dataset into traing and test ser
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=100)
#Apply satndard scalar to time and amount column to bring into the same wight of all datas
#from sklearn.preprocessing import StanardScaler
#sc_X = StandardScaler()
#Sc_X.fit_transform()
#Apply Random Forest Classifier algorithm
#Feature scaling is not needed for RandomForestClassifier as it implemented internally.
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=100)
classifier.fit(X_train, y_train)

#Predict the test result
y_pred = classifier.predict(X_test)
#Confusion Metrics
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred)

cf
#The total is 56866+96 = 56962 - out of 56962 test data - the prediction is 100% correct as you can
#see the credit card transaction having no fraud is 56866 and fraud is 96
#the other values in the diaginal is zero. So there is no incorrect prediction. The y_pred is 
#same as X_test
len(X_test)
