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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
%matplotlib inline
import xgboost

with open('../input/house-prices-advanced-regression-techniques/data_description.txt','r') as f:
    info = f.read()
print(info)

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Quantitative Features: ",quantitative)
print()
print("Qualitative Attributes: ",qualitative)
df.head()
print(df.info())
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
plt.style.use('seaborn')
missing = df.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
bars = missing.plot.bar(missing,color='aqua',edgecolor='black',linewidth=1)
plt.title("Missing Values")
plt.ylabel('Counts')
plt.xlabel('features')
plt.show()

columns_to_drop = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id']
df['SalePrice'].describe()

plt.figure(figsize=(10,6))
sns.distplot(df['SalePrice'],fit=st.norm)
sns.distplot(df['SalePrice'],fit=st.lognorm)
plt.show()
print('Skewness= {:.3f}'.format(df['SalePrice'].skew()))
print('Kurtosis= {:.3f}'.format(df['SalePrice'].kurt()))
df['LogSalePrice'] = np.log1p(df['SalePrice'])
plt.figure(figsize=(10,6))
sns.distplot(df['LogSalePrice'],fit=st.norm,kde=False)
sns.distplot(df['LogSalePrice'],fit=st.lognorm,kde=False)
plt.show()
def relation_with_numerical_feature(VarName,limit):
    data = pd.concat([df['SalePrice'],df[VarName]],axis=1)
    data.plot.scatter(x=VarName,y = 'SalePrice',ylim=(0,limit))
    plt.show()
relation_with_numerical_feature('GrLivArea',900000)
relation_with_numerical_feature("TotalBsmtSF",900000)
f = pd.melt(df, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value",kde_kws={'bw':1})
for c in qualitative:
    df[c] = df[c].astype('category')
    if df[c].isnull().any():
        df[c] = df[c].cat.add_categories(['Missing'])
        df[c] = df[c].fillna('Missing')
        
def boxplot(x,y,**kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)
    
f = pd.melt(df,id_vars=['SalePrice'],value_vars = qualitative)
g = sns.FacetGrid(f,col='variable',col_wrap=2, sharex=False, sharey=False, size=5)
g.map(boxplot,'value','SalePrice')
plt.show()


def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for clas in frame[c].unique():
            s = frame[frame[c] == clas]['SalePrice'].values
            samples.append(s)
        pval = st.f_oneway(*samples)[1]
        pvals.append(pval)
        
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(df)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='features', y='disparity')
x=plt.xticks(rotation=90)
train = df.drop(columns_to_drop,axis=1)
test = test_data.drop(columns_to_drop,axis=1)

sns.heatmap(train.isnull(),cbar=False)
def fill_nan_train(feature):
    if train[feature].isnull().sum()>0:
        if feature in qualitative:
            train[feature].fillna(train[feature].mode()[0],inplace=True)
        elif feature in quantitative:
            train[feature].fillna(train[feature].mean(),inplace=True)
columns_with_nan_values = ['LotFrontage','GarageFinish','GarageType','GarageCond','GarageQual','GarageYrBlt','BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond',
                          'BsmtQual','MasVnrType','Electrical','MasVnrArea','MSZoning','BsmtFullBath','Utilities','BsmtHalfBath','Functional','TotalBsmtSF','GarageArea','BsmtFinSF2','BsmtUnfSF','SaleType','Exterior2nd','Exterior1st','KitchenQual','GarageCars','BsmtFinSF1']
for col in columns_with_nan_values:
    fill_nan_train(col)
train.isnull().sum().sort_values(ascending=False)
print(qualitative,len(qualitative),quantitative,len(quantitative),sep='\n')

sns.heatmap(test.isnull(),cbar=False,cmap = 'ocean')
def fill_nan_test(feature):
    if test[feature].isnull().sum()>0:
        if feature in qualitative:
            test[feature].fillna(test[feature].mode()[0],inplace=True)
        elif feature in quantitative:
            test[feature].fillna(test[feature].mean(),inplace=True)
for col in columns_with_nan_values:
    fill_nan_test(col)
print(test.isnull().sum().sort_values(ascending=False).head(12))
print(test.shape)
print(train.shape)
for i in columns_to_drop:
    if i in qualitative:
        qualitative.remove(i)
    elif i in quantitative:
        quantitative.remove(i)
df_complete = pd.concat([train,test],axis=0)
print('train_shape',train.shape)
print('test_shape',test.shape)
print('df_complete shape',df_complete.shape)
final_df = df_complete.copy()
def category_onehot(features):
    new_df = df_complete
    num = 0
    print(new_df.shape)
    for col in features:
        onehot = pd.get_dummies(new_df[col],drop_first=True)
        new_df.drop(col,axis=1,inplace=True) #drop entire column
        new_df = pd.concat([new_df,onehot],axis=1)
        num += onehot.shape[1]
        print(col)
    print(new_df.shape)
    print(num)
    return new_df
    
new_df = category_onehot(qualitative)
new_df = new_df.loc[:,~new_df.columns.duplicated()]
new_df.shape
X_train = new_df[:1460]
X_test = new_df[1460:]
Y_train = X_train[['SalePrice','LogSalePrice']]

X_train = X_train.drop(['SalePrice','LogSalePrice'],axis=1)
X_test = X_test.drop(['SalePrice','LogSalePrice'],axis=1)
Y_train1 = Y_train['LogSalePrice']
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('Y_train',Y_train.shape)
classifier = xgboost.XGBRegressor()
classifier.fit(X_train,Y_train1)

result = np.exp(classifier.predict(X_test))-1
id_column = test_data['Id']
result = result.reshape((-1,1))
result = pd.DataFrame(result,columns=['SalePrice'])
prediction = pd.concat([id_column,result],axis=1)

#prediction.to_csv('submisison.csv',index=False,header=True)

