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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df.head()
df.drop(columns='Serial No.', inplace=True)
df.head()
df.columns = ['GRE','TOEFL','Rating','SOP','LOR','CGPA','Research','Chance']
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df.describe()
df.isnull().sum()
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,5))
ax1.set_title("Admission Chance vs GRE Score")
ax2.set_title("Admission Chance vs TOEFL Score")
sns.regplot(x=df.GRE, y=df.Chance, ax=ax1)
sns.regplot(x=df.TOEFL, y=df.Chance, ax=ax2)

f,(ax1,ax2) = plt.subplots(1,2, figsize=(20,5))
ax1.set_title("Admission Chance vs GRE (Research)")
ax2.set_title("Admission Chance vs TOEFL (Research)")
sns.scatterplot(x=df.GRE, y=df.Chance, hue=df.Research, ax=ax1)
sns.scatterplot(x=df.TOEFL, y=df.Chance, hue=df.Research, ax=ax2)
sns.lmplot(x='GRE', y='Chance', hue='Research', data=df)
plt.title("Admission Chance vs GRE Score (Research)")
sns.lmplot(x='TOEFL', y='Chance', hue='Research', data=df)
plt.title("Admission Chance vs TOEFL Score (Research)")
sns.lmplot(x='CGPA', y='Chance', hue='Research', data=df)
plt.title("Admission Chance vs CGPA")
f,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,5))
ax1.set_title("Admission Chance vs LOR")
ax2.set_title("Admission Chance vs SOP")
ax3.set_title("Admission Chance vs School Rating")
sns.scatterplot(x=df.LOR, y=df.Chance,ax=ax1)
sns.scatterplot(x=df.SOP, y=df.Chance,ax=ax2)
sns.scatterplot(x=df.Rating, y=df.Chance, ax=ax3)

features=['GRE','TOEFL','CGPA','Research']
X=df[features]
y=df['Chance']
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_hat=lr.predict(X_test)
print("R2: ",lr.score(X_test,y_test))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_hat)))