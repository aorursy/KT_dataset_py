# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = '/kaggle/input/pulsar-star/pulsar_stars.csv'
df = pd.read_csv(data)
df.head()
df['target_class'].value_counts()
df.info() #check for datatypes and counts.
#do we need to normalize the features? 
#to answer this question we can check the output of describe()
round(df.describe(),2)
#features
X = df.drop(['target_class'],axis=1) # 0 is rows and 1 is column, we are dropping the entire column
#target variable
y = df['target_class']
X.head()
#now we will normalize the features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #initialize a standard scaler
X = scaler.fit_transform(X) #use it to normalize the data

X[1:2,:] # compare this output against the above non-normalized data

#also note that now X is a numpy array whereas previously it was a dataframe.
df.boxplot(column='IP Mean',figsize=(8,8))
#in the below plot box's boundaries are Q1 and Q3, The middle line in the box is Q2.
#The Black horizontal lines outside the box are minimum and maximum as defined earlier.
#the continous overlapping black circles are outliers.
plt.figure(figsize=(18,10))
df.boxplot()
#we can use separate subplots for better view
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#note first X are assigned then y are assigned in the assignment statement
X_train.shape , X_test.shape
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
from sklearn.metrics import accuracy_score
print('Model accuracy without C:',accuracy_score(y_test,y_pred))
svc = SVC(C=100)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print('Model accuracy with C=100:',accuracy_score(y_test,y_pred))
svc = SVC(C=1000)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print('Model accuracy with C=100:',accuracy_score(y_test,y_pred))