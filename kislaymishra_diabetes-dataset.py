# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
des=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
des
##obtaining statistics of the above dataset
des.describe()
m=des[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin']].replace(0,np.nan)
##filling missing values with the means of their respective columns
m.fillna(des.mean(),inplace=True)

m
m.isnull().sum()
##Join the columns in one dataframe 
m1=des['BMI']
m1
m2=des['DiabetesPedigreeFunction']
m2
m3=des['Age']
m3
m4=des['Outcome']
m4
ds=pd.concat([m,m1,m2,m3,m4],axis=1)
ds
##checking the null values
ds.isnull().sum()
##Again obtaining statistics of the new dataset with no missing values
ds.describe()
##checking of null values 
sns.heatmap(ds.isnull(),cmap='viridis')
##obtaining correlation matrix of the data
ds.corr()
## Visualisation of the correlation matrix 
plt.figure(figsize=(10,8))
sns.heatmap(ds.corr(),cmap='coolwarm',annot=True)
sns.pairplot(ds)
ds['Glucose'].hist(bins=20)
ds['BloodPressure'].hist()
##Training The model using Logistics Regression 
##droping the outcome column from the rest of the columns 
x=ds.drop(['Outcome'],axis=1)
x
y=ds['Outcome']
y
from sklearn.model_selection import train_test_split
##spliting the dataset into train and test set data
X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.30, random_state=101)
X_train.shape
X_test.shape
##importing Logistic regression model from the scikit learn library of the python
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
##fitting of the model
model.fit(X_train,y_train)
##predicting the model
predictions=model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
##Obtaining confusion matrix to get corrected and incorrected values
confusion_matrix(y_test,predictions)
##classification report give overall accuracy of the model .It gives precision values ,f1 score and accuracy
#the model has 77% percent accuracy 
print(classification_report(y_test,predictions))
