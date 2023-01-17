import pandas as pd

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.decomposition import PCA

from sklearn import preprocessing



from IPython.display import Image

#import prince

from IPython.core.display import HTML 



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df = data.copy()

df
df['SalePrice'].describe()
sns.distplot(df['SalePrice']) #May be some people can afford expensive houses 'Outliers'
print("Skewness: %f" % df['SalePrice'].skew())

print("Kurtosis: %f" % df['SalePrice'].kurt())
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df[cols], size = 2.5)

plt.show();
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.isnull().sum()
Pool = {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" :3, "Gd" : 4, "Ex" : 5}

df['PoolQC'] = df['PoolQC'].map(Pool)

test['PoolQC'] = test['PoolQC'].map(Pool) #Trying mapping of Categorical variable. it will be discarded after cleaning

#Data imputation/cleaning

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25) 
step1 = missing_data[(missing_data['Percent'] > .20)]

step1
len(df), len(df.columns)
df = df.drop(step1.index, 1)
len(df), len(df.columns)
step2 = missing_data[(missing_data['Percent'] <= .20) & (missing_data['Percent'] > 0.10)]

step2
df = df.dropna(subset=step2.index)
len(df), len(df.columns)
step3 = missing_data[(missing_data['Percent'] <= .10) & (missing_data['Percent'] > 0)]

step3
df[step3.index] = df[step3.index].fillna(df.mode().iloc[0])
len(df), len(df.columns)
df.isnull().sum().max()
df2 = data.transpose() #data.T
df2
total = df2.isnull().sum().sort_values(ascending=False)

percent = (df2.isnull().sum()/df2.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
step1 = missing_data[(missing_data['Percent'] > .20)]

step1
step2 = missing_data[(missing_data['Percent'] <= .20) & (missing_data['Percent'] > 0.10)]

step2
df2 = df2.dropna(subset=step2.index)
len(df), len(df.columns)
step3 = missing_data[(missing_data['Percent'] <= .10) & (missing_data['Percent'] > 0)]

step3
df2[step3.index] = df2[step3.index].fillna(df2.mode().iloc[0])
df.isnull().sum().max()
for col in df.columns:

    

    fig, ax = plt.subplots(figsize=(8,4))

    

    ax.scatter(df['SalePrice'], df[col])

    

    ax.set_xlabel('SalePrice')

    ax.set_ylabel(col)

    plt.title('SalePrice & %s'%col)

    plt.show()
numerical   = df.select_dtypes(exclude=['object'])

categorical = df.select_dtypes(include=['object'])
numerical.columns
#selecting only top numerical features

corrmat =numerical.corr()

f, ax = plt.subplots(figsize=(12, 9))

# sns.heatmap(corrmat, vmax=.8, square=True)



#saleprice correlation matrix

k=15  #top 15 variable corr

cols= corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm= np.corrcoef(numerical[cols].values.T)

sns.set(font_scale=1.25)

hm= sns.heatmap(cm, cbar=True, annot=True, square=True,

                fmt='.2f', annot_kws={'size':10},

                yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
#highest correlation b/w these two

ax = sns.boxplot(x="OverallQual", y="SalePrice", data=df)
plt.bar(df['MoSold'],df['SalePrice'])

plt.xlabel('Categories')

plt.ylabel("Months")

plt.title('SalePrice')

plt.show()
plt.bar(df['YrSold'],df['SalePrice'])

plt.xlabel('Categories')

plt.ylabel("Year")

plt.title('SalePrice')

plt.show()
#MSSubClass={"20":'1 story 1946+', "30":'1 story 1945-', "40":'1 story unf attic', "45":'1,5 story unf', "50":'1,5 story fin', "60":'2 story 1946+', "70":'2 story 1945-', "75":'2,5 story all ages', "80":'split/multi level', "85":'split foyer', "90":'duplex all style/age', "120":'1 story PUD 1946+', "150":'1,5 story PUD all', "160":'2 story PUD 1946+', "180":'PUD multilevel', "190":'2 family conversion'}

#df['MSSubClass'] = df['MSSubClass'].map(MSSubClass)

#test['MSSubClass'] = test['MSSubClass'].map(MSSubClass)
#correlation again

corrmat =df.corr()

f, ax = plt.subplots(figsize=(12, 9))

# sns.heatmap(corrmat, vmax=.8, square=True)



#saleprice correlation matrix

k=15

cols= corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm= np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm= sns.heatmap(cm, cbar=True, annot=True, square=True,

                fmt='.2f', annot_kws={'size':10},

                yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
plt.bar(df['Neighborhood'],df['SalePrice'])

plt.xlabel('SalePrice')

plt.ylabel('Neighborhood')

plt.title('b/w 2')

plt.figure(figsize=(1000,800))

plt.show()
#Mapping of variables

Quality={"Ex":"5", "Gd":"4" ,"TA":"3", "Fa":"2", "Po":"1","NA":"0"}

df['ExterCond'] = df['ExterCond'].map(Quality)

test['ExterCond'] = test['ExterCond'].map(Quality)
df['ExterQual'] = df['ExterQual'].map(Quality)

test['ExterQual'] = test['ExterQual'].map(Quality)
df['BsmtQual'] = df['BsmtQual'].map(Quality)

test['BsmtQual'] = test['BsmtQual'].map(Quality)
df['BsmtCond'] = df['BsmtCond'].map(Quality)

test['BsmtCond'] = test['BsmtCond'].map(Quality)
df['HeatingQC'] = df['HeatingQC'].map(Quality)

test['HeatingQC'] = test['HeatingQC'].map(Quality)
df['KitchenQual'] = df['KitchenQual'].map(Quality)

test['KitchenQual'] = test['KitchenQual'].map(Quality)
df['GarageQual'] = df['GarageQual'].map(Quality)

test['GarageQual'] = test['GarageQual'].map(Quality)
df['3SsnPorch'].sum()

df.drop('3SsnPorch',axis=1)
df['GarageCond'] = df['GarageCond'].map(Quality)

test['GarageCond'] = test['GarageCond'].map(Quality)
numerical   = df.select_dtypes(exclude=['object'])

numerical.columns


#Qual=[df['ExterQual'],df['BsmtQual'],df['ExterCond'],df['BsmtCond'],df['HeatingQC'],df['KitchenQual'],df['GarageQual'],df['GarageCond']]

#mca=prince.MCA()

#mca = mca.fit(Qual) # same as calling ca.fs_r(1)

#mca = mca.transform(Qual) # same as calling ca.fs_r_sup(df_new) for *another* test set.

#print(Qual)

TotalBath=df['BsmtFullBath']+df['BsmtHalfBath']+df['FullBath']+df['HalfBath']
from scipy.stats import pearsonr

corr, _ = pearsonr(TotalBath,df['SalePrice'])

print('Pearsons correlation: %.3f' % corr)
plt.figure(figsize=(8,4))

plt.hist(df['SalePrice'], bins=30, alpha=0.5, label="SalePrice")

plt.hist(TotalBath, bins=10, alpha=0.5, label="TotalBath")
Renowate=2020-df['YearRemodAdd']

from scipy.stats import pearsonr

corr, _ = pearsonr(Renowate,df['SalePrice'])

print('Pearsons correlation: %.3f' % corr)
df=df.drop('BsmtFullBath',axis=1)

df=df.drop('BsmtHalfBath',axis=1)

df=df.drop('FullBath',axis=1)

df=df.drop('HalfBath',axis=1)
df['Renowate'] = Renowate

df['TotalBath'] = TotalBath
df=df.drop('YearRemodAdd',axis=1)

df=df.drop('YearBuilt',axis=1)
df
plt.figure(figsize=(8,4))

plt.hist(df['SalePrice'], bins=25, alpha=0.5, label="SalePrice")

plt.hist(df['Neighborhood'], bins=25, alpha=0.5, label="Neighborhood")
TotalArea=df['GrLivArea'] +df['TotalBsmtSF']

df['TotalBsmtSF'].isnull().sum()
from scipy.stats import pearsonr

corr, _ = pearsonr(TotalArea,df['SalePrice'])

print('Pearsons correlation: %.3f' % corr)
df['TotalArea']=TotalArea
df=df.drop('GrLivArea',axis=1)

df=df.drop('TotalBsmtSF',axis=1)
Porch=df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
corr, _ = pearsonr(Porch,df['SalePrice'])

print('Pearsons correlation: %.3f' % corr)
df=df.drop('BsmtFinSF1',axis=1)

df=df.drop('GarageYrBlt',axis=1)

df=df.drop('GarageCond',axis=1)

df=df.drop('GarageArea',axis=1)
corr, _ = pearsonr(df['OpenPorchSF'],df['SalePrice'])

print('Pearsons correlation: %.3f' % corr)
df=df.drop('EnclosedPorch',axis=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categorical = df.select_dtypes(include=['object'])

def preprocess(df, str_labels):

    for label in str_labels:

        le.fit(df[label])

        df[label] = le.transform(df[label])

    return df



df = preprocess(df, categorical.columns)
categorical = df.select_dtypes(include=['object'])
categorical.columns #check if anything left in category
numerical = df.select_dtypes(exclude=['object'])

from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

for col in numerical.columns:

    df[col]=scaler.fit_transform(numerical[col].values.reshape(-1, 1))
Q1 = numerical.quantile(0.25)

Q3 = numerical.quantile(0.75)

IQR = Q3 - Q1
threshold = 10

iqrdf = (numerical < (Q1 - threshold * IQR)) | (numerical > (Q3 + threshold * IQR))
numerical = numerical[~iqrdf.any(axis=1)]

categorical = categorical[~iqrdf.any(axis=1)]
categorical.shape, numerical.shape
df = pd.concat([categorical, numerical], axis=1)

df
df['PoolArea'].sum()

df=df.drop('PoolArea',axis=1)
df['3SsnPorch'].sum()

df=df.drop('3SsnPorch',axis=1)
df['ScreenPorch'].sum()

df=df.drop('ScreenPorch',axis=1)
df['MiscVal'].sum()

df=df.drop('MiscVal',axis=1)
df=df.drop('SaleType',axis=1)
df=df.drop('Id',axis=1)
df['BsmtFinSF2'].sum()

df=df.drop('BsmtFinSF2',axis=1)
df=df.drop('SaleCondition',axis=1)
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import KBinsDiscretizer



from sklearn.model_selection import train_test_split 

from sklearn import metrics



from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression



X=df.drop('SalePrice',axis=1)

enc = KBinsDiscretizer(n_bins=50, encode='ordinal')

y = enc.fit_transform(np.array(df['SalePrice']).reshape(-1, 1))



lin_reg=LinearRegression()

MSEs=cross_val_score(lin_reg,X,y,scoring='neg_mean_squared_error',cv=5)

mean=np.mean(MSEs)

print(mean)
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso
ridge=Ridge()

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_) #need to reverse the log to the real values
Z=df.drop(["SalePrice"],axis=1) # Our samples

Z=Z.drop(["GarageFinish"],axis=1) # Our samples

Z=Z.drop(["Exterior2nd"],axis=1)

Z=Z.drop(["BsmtFinType1"],axis=1)

Z=Z.drop(["BsmtUnfSF"],axis=1) #this is perfect





Z.columns
from sklearn.model_selection import train_test_split



Z_train,Z_test,y_train,y_test= train_test_split(Z,y,test_size=0.2)
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split 

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

np.random.seed(42)



rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(Z_train,y_train)

rfc_preds= rfc.predict(Z_test)



accuracy_score(y_test,rfc_preds)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(Z_train, y_train)

gbk_preds = gbk.predict(Z_test)

accuracy_score(y_test,gbk_preds)