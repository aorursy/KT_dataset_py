# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/BlackFriday.csv")
data.head()
data.shape
print(data.isnull().sum())
data.fillna(0.0,inplace=True)
data.head()
import seaborn as sns

sns.countplot(data['Marital_Status'])

sns.countplot(data['Gender'])
data['Age'].value_counts().plot.bar()
sns.countplot(data['Product_Category_1'],hue=data['Gender'])
sns.countplot(data['Product_Category_1'],hue=data['Marital_Status'])
sns.countplot(data['Product_Category_2'],hue=data['Gender'])
sns.countplot(data['Product_Category_3'],hue=data['Gender'])
data['Occupation'].value_counts().plot.bar()
sns.countplot(data['Occupation'],hue=data['Gender'])
sns.countplot(data['City_Category'])
one_hot = pd.get_dummies(data['City_Category'])

# Drop column B as it is now encoded

data = data.drop('City_Category',axis = 1)

# Join the encoded df

df = data.join(one_hot)
df.head()
df['Gender'] = np.where(df['Gender']=='M',1,0)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
df.drop(['Product_ID','User_ID'],axis=1,inplace=True)
df_Age = pd.get_dummies(df.Age)



df_SIC = pd.get_dummies(df.Stay_In_Current_City_Years)

df_encoded = pd.concat([df,df_Age,df_SIC],axis=1)

df_encoded.drop(['Age','Stay_In_Current_City_Years'],axis=1,inplace=True)
df_frac = df_encoded.sample(frac=0.02,random_state=100)

X = df_frac.drop(['Purchase'], axis=1)

y = df_frac['Purchase']

scaler = StandardScaler().fit(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=100)

param_grid = {'n_estimators':[10,30,100,150],'max_depth':[5,7,9]}

grid_rf = GridSearchCV(RandomForestRegressor(),param_grid,cv=3,scoring='neg_mean_squared_error').fit(X_train,y_train)
print('Best parameter: {}'.format(grid_rf.best_params_))

print('Best score: {:.2f}'.format((-1*grid_rf.best_score_)**0.5))
rf = RandomForestRegressor(max_depth=7, n_estimators=30).fit(X_train,y_train)

f_im = rf.feature_importances_.round(3)

ser_rank = pd.Series(f_im,index=X.columns).sort_values(ascending=False)



plt.figure()

sns.barplot(y=ser_rank.index,x=ser_rank.values,palette='deep')

plt.xlabel('relative importance')
X = df_encoded.drop(['Purchase'], axis=1)

y = df_encoded['Purchase']

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=100)







rf = RandomForestRegressor(max_depth=7, n_estimators=150).fit(X_train,y_train)

y_predict = rf.predict(X_test)

print('Test set RMSE: {:.3f}'.format(mean_squared_error(y_test,y_predict)**0.5))
y_predict_train = rf.predict(X_train)

print('Train set RMSE: {:.3f}'.format(mean_squared_error(y_train,y_predict_train)**0.5))
y_predict_train = rf.predict(X_train)

print('Train set RMSE: {:.3f}'.format(mean_squared_error(y_train,y_predict_train)**0.5))