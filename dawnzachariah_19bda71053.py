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
%matplotlib inline
data=pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')
test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')
sample=pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
data.shape
data.dtypes
data.columns
data.nunique()
data.head()
data.tail()
data.info()
data.flag.value_counts()
data.isnull().sum()
data.isnull()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.pairplot(data,hue='flag')
#obtaining the correlation matrix using heatmap
corrmat = data.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
data.describe()
plt.figure(figsize=(12, 5))
sns.boxplot(y='currentBack',data=data,palette='winter')
plt.figure(figsize=(12, 5))
sns.boxplot(y='positionBack',data=data,palette='winter')
plt.figure(figsize=(12, 5))
sns.boxplot(y='refPositionBack',data=data,palette='winter')
plt.figure(figsize=(12, 5))
sns.boxplot(y='refVelocityBack',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='velocityBack',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='currentFront',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='positionFront',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='refPositionFront',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='refVelocityFront',data=data,palette='winter')
plt.figure(figsize=(15, 5))
sns.boxplot(y='velocityFront',data=data,palette='winter')
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
print(data_out.shape)
data["currentBack"] = data["currentBack"].map(lambda i: np.log(i) if i > 0 else 0) 
test["currentBack"] = test["currentBack"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['currentBack'].skew())
data["positionBack"] = data["positionBack"].map(lambda i: np.log(i) if i > 0 else 0) 
test["positionBack"] = test["positionBack"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['positionBack'].skew())
data["refPositionBack"] = data["refPositionBack"].map(lambda i: np.log(i) if i > 0 else 0) 
test["refPositionBack"] = test["refPositionBack"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['refPositionBack'].skew())
data["refVelocityBack"] = data["refVelocityBack"].map(lambda i: np.log(i) if i > 0 else 0) 
test["refVelocityBack"] = test["refVelocityBack"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['refVelocityBack'].skew())
data["velocityBack"] = data["velocityBack"].map(lambda i: np.log(i) if i > 0 else 0) 
test["velocityBack"] = test["velocityBack"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['velocityBack'].skew())
data["currentFront"] = data["currentFront"].map(lambda i: np.log(i) if i > 0 else 0) 
test["currentFront"] = test["currentFront"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['currentFront'].skew())
data["positionFront"] = data["positionFront"].map(lambda i: np.log(i) if i > 0 else 0) 
test["positionFront"] = test["positionFront"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['positionFront'].skew())
data["refPositionFront"] = data["refPositionFront"].map(lambda i: np.log(i) if i > 0 else 0) 
test["refPositionFront"] = test["refPositionFront"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['refPositionFront'].skew())
data["refVelocityFront"] = data["refVelocityFront"].map(lambda i: np.log(i) if i > 0 else 0) 
test["refVelocityFront"] = test["refVelocityFront"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['refVelocityFront'].skew())
data["velocityFront"] = data["velocityFront"].map(lambda i: np.log(i) if i > 0 else 0) 
test["velocityFront"] = test["velocityFront"].map(lambda i: np.log(i) if i > 0 else 0) 
print(data['velocityFront'].skew())
data.describe()
X = data.drop('flag', axis=1) 
y = data[['flag']]
#Splitting Independent variables as X_train & X_test and Dependent variable as y_train & y_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train.head()
y_train.head()
X_test.head()
y_test.head()
#importing RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train,y_train)
#predicting for X_test
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#predicting for Test_Mask_Dataset.csv
pred=model.predict(test)
pred
#attaching the predicted value
sample['flag']=pred
sample.head()
#wrting to the csv Sample Submission.csv
sample.to_csv('Sample Submission.csv', index=False)