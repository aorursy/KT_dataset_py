

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.base import TransformerMixin

import warnings

warnings.filterwarnings("ignore")
#getting training and test data

train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#just taking a look at our data

train.head(10)

train.info()
#describing basics

train.describe()
sns.distplot(train['SalePrice'])

plt.xticks(rotation=30)
#skewness and kurtosis

print("Skewness:{}".format(train['SalePrice'].skew()))

print("Kurtosis:{}".format(train['SalePrice'].kurt()))
train['SalePrice']=np.log(train['SalePrice'])

sns.distplot(train['SalePrice'])
print("Skewness:{}".format(train['SalePrice'].skew()))

print('kurtosis:{}'.format(train['SalePrice'].kurt()))


sns.scatterplot(data=train,x='GrLivArea',y='SalePrice')

sns.scatterplot(data=train,x='GarageArea',y='SalePrice')
categorical=['SaleCondition','YrSold','OverallQual','LotShape','CentralAir','HouseStyle']

fig,ax=plt.subplots(2,3,figsize=(15,10))

for var,subplot in zip(categorical,ax.flatten()):

    sns.boxplot(data=train,x=var,y='SalePrice',ax=subplot)

plt.tight_layout()    
#correlation

corr=train.corr()

fig,ax=plt.subplots(figsize=(15,10))

sns.heatmap(corr)
n_large=corr.nlargest(10,'SalePrice')['SalePrice']

n_large.plot.bar()
#fucntion to get missing percentage value

def get_perctg(data):

    nan=data.isna().sum()/data.shape[0]

    perctg=nan*100

    return perctg[perctg>0]

get_perctg(train)
thresh=np.ceil((20/100)*train.shape[0])

train=train.dropna(thresh=thresh,how='any',axis=1)
class Impute(TransformerMixin):

    def __init__(self):

        """"impute missing categorical as well as

        numericla """

    def fit(self,X,y=None):

        self.fill=pd.Series([X[c].value_counts().index[0]

                           if X[c].dtype==np.dtype('O') else X[c].mean() for c in X],

                            index=X.columns)

        return self

    def transform(self,X,y=None):

        return X.fillna(self.fill)

                            

                            

    



imputer=Impute()



train=imputer.fit_transform(train)