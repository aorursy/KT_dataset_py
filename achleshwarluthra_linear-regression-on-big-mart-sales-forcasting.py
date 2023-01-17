import pandas as pd
df = pd.read_csv('/kaggle/input/big-mart-sales-forcasting/train.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
nulldf = df[df['Item_Weight'].isnull()]
nulldf.head()
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df.isnull().sum()
df['Outlet_Size'].value_counts()
df['Outlet_Size'] = df['Outlet_Size'].fillna('NA')
df.isnull().sum()
df.describe()
a = df.groupby('Item_Fat_Content', as_index=False)['Item_Identifier'].count()
a
import numpy as np
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='LF', 'Low Fat', df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='low fat', 'Low Fat', df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='reg', 'Regular', df['Item_Fat_Content'])
df['Count'] = 1
Fat_Content = df.groupby('Item_Fat_Content', as_index=False)['Count'].sum()
Fat_Content
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(x = 'Item_Fat_Content', y = 'Count', data=Fat_Content)
plt.show()
item_type = df.groupby('Item_Type', as_index=False)['Count'].count()
item_type.sort_values('Count', ascending=False)
ax2 = sns.barplot(x = 'Item_Type', y = 'Count', data = item_type)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90)
plt.show()
ax3 = sns.distplot(df['Item_MRP'])
ax3 = sns.distplot(df['Item_MRP'], bins=10)
box = sns.boxplot(df['Item_MRP'])
box2 = sns.boxplot(df['Item_Outlet_Sales'])
from scipy.stats import iqr

q1 = df['Item_Outlet_Sales'].quantile(0.25)
q3 = df['Item_Outlet_Sales'].quantile(0.75)
inter_qr = iqr(df['Item_Outlet_Sales'])
print(q1)
print(q3)
print(inter_qr)
df['Outliers'] = 0
df['Outliers'] = np.where(df['Item_Outlet_Sales']>(q3+1.5*inter_qr), 1, df['Outliers'])
df['Outliers'] = np.where(df['Item_Outlet_Sales']<(q1-1.5*inter_qr), 1, df['Outliers'])
df.head()
box2 = sns.boxplot(df['Item_Visibility'])
vq1 = df['Item_Visibility'].quantile(0.25)
vq3 = df['Item_Visibility'].quantile(0.75)
vinter_qr = iqr(df['Item_Visibility'])
print(vq1)
print(vq3)
print(vinter_qr)
df['Outliers'] = np.where(df['Item_Visibility']>(vq3+1.5*vinter_qr), 1, df['Outliers'])
df['Outliers'] = np.where(df['Item_Visibility']<(vq1-1.5*vinter_qr), 1, df['Outliers'])
df.head()
df.Outliers.value_counts()
df.Outlet_Establishment_Year.describe()
df['Outlet_Age'] = 2015 - df['Outlet_Establishment_Year']
df.head()
df['Outlet_Age'].describe()
dfclean = df[df['Outliers']==0]
print(df.shape)
print(dfclean.shape)
dfclean = dfclean.drop(columns=['Item_Identifier', 'Count', 'Outliers'])
dfclean.shape
dfclean = pd.get_dummies(dfclean)
dfclean.shape
dfclean.head()
x = dfclean.drop(columns='Item_Outlet_Sales')
y = dfclean['Item_Outlet_Sales']

print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
import statsmodels.api as sm

model = sm.OLS(ytrain, xtrain).fit()
print(model.summary())
pred = model.predict(xtest)

data = list(zip(ytest, pred))
comptab = pd.DataFrame(data, columns=['Actual', 'Predicted'])
comptab.head()
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(ytest, pred))
print(rmse)
