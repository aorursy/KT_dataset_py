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
import missingno

%matplotlib inline

#Reading the dataframe

df = pd.read_csv("/kaggle/input/hr-analytics/HR_comma_sep.csv")
df.head()

#checking the shape of the data 

df.shape
#Checking Statistical Overview of Data

df.describe(include='all')

# Checking Unique values of Department

df['Department'].unique()
# Checking Unique values of Salary

df['salary'].unique()
# Plot graphic of missing values

missingno.matrix(df, figsize = (30,10))
plt.show()

#Checking with isnull() function 

df.isnull().any()
#Counting Unique Values of Department.

df['Department'].value_counts()
#Analysing Department Column

df.groupby('Department').sum()
#Getting Unique values of the Left column

df.groupby('left').count()['satisfaction_level']
# Comparision of who left the company based on Salary and Department they are working in.

fig,axis=plt.subplots(nrows=2,ncols=1,figsize=(12,10))
sns.countplot(x='salary',hue='left',data=df,ax=axis[0])
sns.countplot(x='Department',hue='left',data=df,ax=axis[1])
plt.show()
# Comparision of who left based on Satisfaction level

plt.figure(figsize=(7,8))
sns.barplot(x='left',y='satisfaction_level',data = df)
plt.show()

#Checking the correlation between the features

corr = df.corr(method='pearson')
corr
#Checking the correlation between the features using Heatmap

sns.heatmap(corr, annot=True)
plt.show()
#CONVERTING CATEGORICAL FEATURES TO NUMERICAL ONE

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

sal_num=le.fit_transform(df['Department'])
dept = le.fit_transform(df['salary'])
df.head()
df['salary_num'] = sal_num
df['dept'] = dept
df.head()
df.drop(['Department','salary'],axis=1,inplace=True)
df.head()
df.head()
X=df.drop(['left'],axis=1)
y=df['left']

#Importing Essential Machine Learning Algorithms to test the efficiency

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf
# Checking the shape of Train and Test dataset 

print("The X training set shape = ",X_train.shape)
print("The X testing set shape = ",X_test.shape)

# K Nearest Neighbors(KNN)

knn = KNeighborsClassifier()
knn.fit(X_test,y_test)

#Accuracy of KNN

(knn.score(X_test,y_test)*100)
cross_val_score(KNeighborsClassifier(), X, y.values.ravel(),cv=3).mean()*100
#Support Vector Machine(SVM)

sv = SVC()
sv.fit(X_test,y_test)
#Accuracy of SVM

sv.score(X_test,y_test)*100
cross_val_score(SVC(gamma='auto'), X, y.values.ravel(),cv=3).mean()*100
# Decision Tree Classifier

dt=DecisionTreeClassifier()
dt.fit(X,y)
#Accuracy of Decision Tree Classifier

dt.score(X_test,y_test)*100
cross_val_score(DecisionTreeClassifier(),X,y,cv=3).mean()*100
