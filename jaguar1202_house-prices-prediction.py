import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv("../input/train.csv")

data_train.head()
data_train['SalePrice'].describe()
pd.set_option('display.max_columns',100)

data_train.describe(include='all')
#sns.distplot(data_train['SalePrice'])
isnull = data_train.isnull()

display(isnull.sum().sort_values())

display(len(isnull))
#data_train里每列有多少个不同的值

nbvalues = data_train.nunique()

nbvalues = nbvalues.sort_values()

nbvalues
def null():

    for i in range(0,len(isnull)):

        if(isnull[i]):

            display()

        else:

            print("false",i )
corrmat = data_train.corr()

f, ax = plt.subplots(figsize=(20,9))

sns.heatmap(corrmat,square=True,cmap='GnBu',vmin=0.5)


k  = 10 # 关系矩阵中将显示10个特征

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index #和salePrice最有关联的前10个属性

# 若没有.index，返回的是float, 否则，返回的是object



corrmat2 = data_train[cols].corr()
sns.heatmap(corrmat2, annot=True,cmap='GnBu')

#annot 是否显示相关率
import sklearn

from sklearn.model_selection import train_test_split



cols = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd',\

        'YearBuilt']

x = data_train[cols].values

y = data_train['SalePrice'].values

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 100)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

import numpy as np



model = DecisionTreeClassifier(criterion = "gini",min_samples_leaf=31)

model2 = RandomForestRegressor(n_estimators=400)
model2.fit(x_train, y_train)

y_pred = model2.predict(x_test)

print("cost:" + str(np.sum(y_pred-y_test)/len(y_pred)))
#plt.plot(y_pred,y_pred)

range=np.arange(0.,10000.,600000)

plt.plot(y_test,y_test,'r--',y_test,y_pred,'bs')



plt.show()
data_test = pd.read_csv("../input/test.csv")

data_test[cols].isnull().sum()
data_test['GarageArea'].describe()
data_test['GarageArea'].mean()
data_test['GarageArea'].fillna(data_test['GarageArea'].mean(),inplace=True)
clfs={

    'GarageCars':data_test['GarageCars'],

    'TotalBsmtSF':data_test['TotalBsmtSF']

}



for clf in clfs:

    try:

        clfs[clf].fillna(clfs[clf].mean(),inplace=True)

        print(clfs[clf].mean())

    except Exception as e:

        print(clf + "Error:")
data_test[cols].isnull().sum()
data_test_x=pd.concat([data_test[cols]],axis=1)

X=data_test_x.values

Y_pred=model2.predict(X)

Y_pred
prediction=pd.DataFrame(Y_pred,columns=['SalePrice'])

result=pd.concat([data_test['Id'],prediction],axis=1)

result.columns
result.to_csv('./Predictions.csv',index=False)