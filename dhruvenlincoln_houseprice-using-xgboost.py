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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

df.head()
df.isnull().sum()
corr = df.corr()

corr['price'].sort_values(ascending=False)
df.drop(['id','zipcode','date'],axis=1,inplace=True)
df.head()
plt.figure(figsize=(19,14))

sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
sns.barplot(data=df,x='bedrooms',y='price')
plt.figure(figsize=(10,5))
plt.subplot(2,1,2)
sns.barplot(data=df,x='grade',y='price',palette='rocket')
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
sns.barplot(data=df,x='floors',y='price')
plt.figure(figsize=(10,5))
plt.subplot(2,1,2)
sns.barplot(data=df,x='condition',y='price',palette='rainbow')
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
sns.barplot(data=df,x='waterfront',y='price')
plt.figure(figsize=(10,5))
plt.subplot(2,1,2)
sns.barplot(data=df,x='view',y='price',palette='vlag')
sns.countplot(data=df,x='waterfront')
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
sns.barplot(data=df,x='sqft_living',y='price')
plt.figure(figsize=(10,5))
plt.subplot(2,1,2)
sns.barplot(data=df,x='sqft_above',y='price')

plt.figure(figsize=(10,5))
sns.distplot(df['price'],kde=False,color='red')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df.columns
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
model = []

score = []
lf = Lasso()
lf.fit(X_train,y_train)
lf_pred = lf.predict(X_test)
print("Score: ",r2_score(lf_pred,y_test))
model.append("Lasso Regression")
score.append(r2_score(lf_pred,y_test))
rf = RandomForestRegressor(n_estimators=100, random_state = 0)
rf.fit(X_train,y_train)
rf_predict = rf.predict(X_test)
print("Score: ",r2_score(rf_predict,y_test))
model.append("Random Forest Regression")
score.append(r2_score(rf_predict,y_test))
xg = XGBRegressor()
xg.fit(X_train,y_train)
xg_predict = xg.predict(X_test)
print("Score: ",r2_score(xg_predict,y_test))
model.append("Xgboost Regression")
score.append(r2_score(xg_predict,y_test))
plt.subplots(figsize=(10, 15))
sns.barplot(y=score,x=model,palette = sns.cubehelix_palette(len(score)))
plt.xlabel("Score")
plt.ylabel("Regression")
plt.title('Regression Score')
plt.show()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(y_test-xg_predict)
plt.title("Xgboost")
#plt.figure(figsize=(5,5))
plt.subplot(1,2,2)
sns.distplot(y_test-lf_pred,color='red')
plt.title("Lasso")
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.scatterplot(y_test,xg_predict)
plt.title("Xgboost")
plt.subplot(2,2,2)
sns.scatterplot(y_test,lf_pred)
plt.title("Lasso")
