# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)
data=pd.read_csv('../input/train.csv')
data.columns
data.head()
data.isnull().sum()
data['SalePrice'].describe()
sns.distplot(data['SalePrice'],vertical=True)
var='GrLivArea'
df=pd.concat([data['SalePrice'],data[var]],axis=1)
df.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
var='TotalBsmtSF'
plt.scatter(var,'SalePrice',data=data)
plt.ylim(0,800000)
plt.show()
data['OverallQual'].describe()
data['OverallQual'].describe()
fig=sns.boxplot(x='OverallQual',y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
corrmat=data.corr()
sns.heatmap(corrmat,vmax=0.8,square=True)
k=10
cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm=np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()


sns.set()
sns.pairplot(data[cols],size=2.5)
plt.show()
total=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percentage],axis=1,keys=['total','percentage'])
missing_data.head(20)
data = data.drop((missing_data[missing_data['total'] > 1]).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)
data.isnull().sum().max()
data.dtypes

main_cols = data.columns
main_cols
data1 = data.drop(columns = ['SalePrice']).columns
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv('../input/train.csv')
train_y = train.SalePrice
train_x = train[data1].select_dtypes(exclude=['object'])
model = RandomForestRegressor()
model.fit(train_x,train_y)
from sklearn.preprocessing import Imputer

test = pd.read_csv('../input/test.csv')
test_x = test[data1].select_dtypes(exclude = ['object'])
my_imputer = Imputer()
test_x = my_imputer.fit_transform(test_x)
predicted_prices = model.predict(test_x)
predicted_prices

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)





