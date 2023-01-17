import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/Train.csv')
df_train.head()
df_train.info()
df_train.describe()
plt.figure(figsize = (10,6))
sns.heatmap(df_train.corr(), annot = True)
sns.distplot(df_train['Item_Outlet_Sales'])
df_train.columns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
ndf_train = (df_train[['Item_Weight', 'Item_Visibility', 'Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales',]])
ndf_train
X = ndf_train.drop(['Item_Outlet_Sales'],axis = 1)
X = imp.fit_transform(X)
X
y = ndf_train['Item_Outlet_Sales']

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 56)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
df_test = pd.read_csv('../input/Test.csv')
df_test
df_test.info()
ndf_test = df_test[['Item_Weight', 'Item_Visibility', 'Item_MRP','Outlet_Establishment_Year']]
ndf_test = imp.fit_transform(ndf_test)
len(ndf_test)
df_test['Item_Outlet_Sales'] = model.predict(ndf_test)
df_submission = df_test[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
df_submission
df_submission.to_csv('Big Market Sales.csv',index=0)
