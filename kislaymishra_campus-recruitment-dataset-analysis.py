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
mt=pd.read_csv('/kaggle/input/datasets_596958_1073629_Placement_Data_Full_Class (1).csv')






mt
#we drop the serial number and slary column from the dataset in order to analyse the data 
rt=mt.drop(['sl_no','salary'],axis=1)
rt
##To get the infromation of the number of columns present in the datasets 
rt.info()
#to obtain specifically columns of the dataset
rt.columns
##To obtain Various statistic of the dataset
rt.describe()
#to check the number of null values in the each columns of the dataset
rt.isnull().sum()
#the graph shows no null value present in the dataset
sns.heatmap(rt.isnull(),cmap='viridis')
#we calculate correlation matrix to get information about the dependecies among the columns
rt.corr()
plt.figure(figsize=(10,8))
sns.heatmap(rt.corr(),cmap='coolwarm',annot=True)
sns.pairplot(rt)
plt.figure(figsize=(10,12))
rt.boxplot()
##To check number of placed students based on their gender through the histogram
sns.countplot(x='status',data=rt,hue='gender')
#To check the number of students placed or not based on their work experience through histogram
sns.countplot(x='status',data=rt,hue='workex')
#To check the number of students placed or not based on their specialisation through Histogram
sns.countplot(x='status',data=rt,hue='specialisation')
#To check the number of students placed or not based on their stream through Histogram
sns.countplot(x='status',data=rt,hue='hsc_s')
#To check the number of students placed or not based on their degree through Histogram
sns.countplot(x='status',data=rt,hue='degree_t')
#To check the number of students placed or not based on their Boards(central or others) through Histogram
sns.countplot(x='status',data=rt,hue='hsc_b')
#to obtain histogram of hsc percentage of students
rt['hsc_p'].plot.hist(bins=15)
#to obtain histogram of ssc percentage of students
rt['ssc_p'].plot.hist(bins=10)
#we use histogram to obtain mba percentage of students
rt['mba_p'].plot.hist(bins=15)
rt['status'].replace('Not Placed',0,inplace=True)
rt
rt['status'].replace('Not Placed',0,inplace=True)
rt
rt['status'].replace('Placed',1,inplace=True)
#To get dummy parameter of column experience through pandas library
workex=pd.get_dummies(rt['workex'],drop_first=True)
#To get dummy parameter of column SSC Board through pandas library
ssc_b=pd.get_dummies(rt['ssc_b'],drop_first=True)
hsc_b=pd.get_dummies(rt['hsc_b'],drop_first=True)
gender=pd.get_dummies(rt['gender'],drop_first=True)
rt=pd.concat([rt,gender,workex,ssc_b,hsc_b],axis=1)
rt
rt.drop(['degree_t','hsc_s'],axis=1,inplace=True)
rt.drop(['specialisation'],axis=1,inplace=True)
rt.drop(['gender'],axis=1,inplace=True)
rt.drop(['hsc_b'],axis=1,inplace=True)
rt.drop(['ssc_b'],axis=1,inplace=True)
rt.drop(['workex'],axis=1,inplace=True)
rt.head(5)
##We now train the model
x=rt.drop(['status'],axis=1)
x
y=rt['status']
y
#from scikit learn we import train test split to obtain train and test set from the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)
#from scikit learn import Logistic regression to fit the model
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
#Now we fit the model
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
predictions
y_test
#from scikit learn library import classifiaction report
from sklearn.metrics import classification_report 
#from scikit learn library import confusion matrix
from sklearn.metrics import confusion_matrix
#to get the confusion matrix of the data
confusion_matrix(y_test,predictions)
#To obtain classification report of the matrix
print(classification_report(y_test,predictions))
##Thank You
