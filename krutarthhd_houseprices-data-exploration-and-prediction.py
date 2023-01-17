import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
%matplotlib inline
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.columns
train['SalePrice'].describe()
plt.figure(figsize=(14,8))
px.histogram(train,x='SalePrice',nbins=80,title='Selling Price Distribution')
plt.figure(figsize=(14,8))
px.scatter(train,x='GrLivArea',y='SalePrice',title='SalePrice vs GrLivArea',render_mode='auto',)
plt.figure(figsize=(14,8))
px.scatter(train,x='TotalBsmtSF',y='SalePrice',title='SalePrice vs TotalBsmtSF',render_mode='auto')
plt.figure(figsize=(14,8))
px.box(train,x='OverallQual',y='SalePrice',title='SalePrice vs OverallQual')
plt.figure(figsize=(20,10))
sns.boxplot(data=train,x='YearBuilt',y='SalePrice')
plt.title('SalePrice vs YearBuilt')
plt.xlabel('Year')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,12))
sns.heatmap(train.corr(),square=True)
#saleprice correlation matrix
plt.figure(figsize=(10,10))
k = 10 #number of variables for heatmap
cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();
total = train.isnull().sum().sort_values(ascending=False)
percent = (total*100) / train.shape[0]
missingData = pd.concat([total,percent],keys=['Total','Percentage'],axis=1)
missingData.head(20)
train.drop((missingData[missingData['Total'] > 1]).index,1,inplace=True)
train=train.dropna()
plt.figure(figsize=(14,8))
px.scatter(train,x='GrLivArea',y='SalePrice',title='SalePrice vs GrLivArea',render_mode='auto',)
train.loc[train['GrLivArea']==4676]
train.loc[train['GrLivArea']==5642]
train = train.drop([1298,523],axis=0)
test = test[train.columns[0:62]]
idTest = pd.DataFrame(test['Id'])
data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)
data=pd.get_dummies(data)
train, test = data[:len(train)], data[len(train):]

X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']

test = test.drop(columns=['SalePrice', 'Id'])
model = XGBRegressor()
model.fit(X,y)
print(r2_score(model.predict(X),y))
