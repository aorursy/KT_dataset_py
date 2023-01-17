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
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.describe()
df['target'].value_counts().plot(kind='bar')
#splitting the dependent and independent variable
y = df['target']#dependet variable
X = df.drop('target',axis=1)
#axis = 0 for dropping rows
#axis = 1 for dropping columns
print(X.head())
print(y.head())
#scaling the feature. It normalises the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X) 
print(X_scaled)#numpy array created for scaled feature
#splitting the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size = 0.3, shuffle =True) 
#shuffle = True 
#shuffles the data into mixed values because we have got the values sorted order
print("The trainig dataset shape:"+str(X_train.shape))
print("The testing dataset shape:"+str(X_test.shape))
print("The target datashape is:"+str(Y_train.shape)+ str(Y_test.shape))
#using logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train, Y_train)#train the model
prediction = lr.predict(X_test)#predict the testdataset
accuracy = accuracy_score(Y_test, prediction) * 100 #calculating the accuracy
print("The accuracy from Logistic Regression is" +str(round(accuracy, 2)))
#using knn 
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors = 5)#the number of neighbors we want to compare
model.fit(X_train, Y_train)
predict1 = model.predict(X_test)
accuracy = accuracy_score(Y_test,predict1)
print("The accuracy from KNN classifier is" +str(round(accuracy, 2)*100))
