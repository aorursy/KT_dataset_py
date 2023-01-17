import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt #Visualization

import seaborn as sns #Visualization

from scipy.stats import norm #Analysis

from sklearn.preprocessing import StandardScaler #Analysis

from scipy import stats #Analysis

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import gc



import missingno as msno



print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print("train data: ", df_train.shape)

print("test data: ", df_test.shape)
df_train.head()
# Check for duplicates

idsUnique = len(set(df_train.id))

idsTotal = df_train.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")



### set: https://www.codingfactory.net/10043
msno.matrix(df_train)
#Missing data check

train_na_ratio = (df_train.isnull().sum() / len(df_train)) * 100

for i in range(np.shape(df_train)[1]):

    print("There are " + str(train_na_ratio[i]) + " ratio of missing data in " + str(df_train.columns[i]) + " variable" )
df_test.head()
# Check for duplicates

idsUnique = len(set(df_test.id))

idsTotal = df_test.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
msno.matrix(df_test)
#Missing data check

test_na_ratio = (df_test.isnull().sum() / len(df_test)) * 100

for i in range(np.shape(df_test)[1]):

    print("There are " + str(test_na_ratio[i]) + " ratio of missing data in " + str(df_test.columns[i]) + " variable" )
df_train['date'] = df_train['date'].apply(lambda x : str(x[:8])).astype(str)

df_test['date'] = df_test['date'].apply(lambda x : str(x[:8])).astype(str)
df_train.date
np.shape(df_train.columns)
np.shape(df_test.columns)
fig, ax = plt.subplots(11, 2, figsize=(20, 60))



count = 0

columns = df_train.columns

for row in range(11):

    for col in range(2):

        sns.kdeplot(df_train[columns[count]], ax=ax[row][col])

        ax[row][col].set_title(columns[count], fontsize=15)

        count+=1

        if count == 21 :

            break
#descriptive statistics summary

df_train.price.describe()
#histogram

plt.figure(figsize=(8, 6))

sns.distplot(df_train['price'])
#skewness and kurtosis

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']), plot=plt)
df_train['price'] = np.log1p(df_train['price'])

#histogram

plt.figure(figsize=(8, 6))

sns.distplot(df_train['price'])
corrmat = df_train.corr()

colormap = plt.cm.RdBu

plt.figure(figsize=(16,14))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(corrmat, fmt='.2f',linewidths=0.1, vmax=0.9, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 10})
print(corrmat.price)
# Find most important features relative to target



corrmat.sort_values(["price"], ascending = False, inplace = True)

print(corrmat.price)
cor_abs = abs(df_train.corr(method='spearman'))

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

colormap = plt.cm.RdBu

plt.title('Spearman Correlation of Features', y=1.05, size=15)

sns.heatmap(cor, fmt='.2f',linewidths=0.1, vmax=0.9, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 8}, xticklabels=cor_cols.values, yticklabels=cor_cols.values)
print(cor_abs.price)
# Find most important features relative to target



cor_abs.sort_values(["price"], ascending = False, inplace = True)

print(cor_abs.price)
# price과 correlation이 높은 변수들

cor_cols
df_train.head(20)
df_train.loc[df_train['sqft_above']/df_train['floors']>df_train['sqft_lot']]
df_train.loc[df_train['id']==2464]['sqft_above']/df_train.loc[df_train['id']==2464]['floors']
df_train.loc[df_train['id']==10987]['sqft_above']/df_train.loc[df_train['id']==10987]['floors']
df_train.loc[df_train['id']==12104]['sqft_above']/df_train.loc[df_train['id']==12104]['floors']
data = pd.concat([df_train['price'], df_train['sqft_lot']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_lot', y="price", data=data, marker="+", color="g")
data = pd.concat([df_train['sqft_living'], df_train['sqft_lot']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_lot', y="sqft_living", data=data, marker="+", color="g")
df_train.loc[(df_train['sqft_lot'] > 1500000)]
data = pd.concat([df_test['sqft_living'], df_test['sqft_lot']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_lot', y="sqft_living", data=data, marker="+", color="g")
df_test.loc[(df_test['sqft_lot'] > 1000000)]
data = pd.concat([df_train['price'], df_train['sqft_lot15']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_lot15', y="price", data=data, marker="+", color="g")
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

plt.figure(figsize=(8, 6))

sns.boxplot(x='grade', y="price", data=data)
sns.set(color_codes=True)

data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_living', y="price", data=data, marker="+", color="g")



# seaborn, regplot: https://seaborn.pydata.org/generated/seaborn.regplot.html
plt.figure(figsize = (8, 5))

sns.jointplot(df_train.sqft_living, df_train.price, 

              alpha = 0.5)

plt.xlabel('sqft_living')

plt.ylabel('price')

plt.show()
data = pd.concat([df_train['price'], df_train['sqft_living15']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_living15', y="price", data=data, marker="+", color="g")
plt.plot(df_train['sqft_living15']-df_train['sqft_living'])
plt.plot(df_train['sqft_lot15']-df_train['sqft_lot'])
df_train.loc[(df_train['sqft_lot15']-df_train['sqft_lot'] < -1200000)]
plt.figure(figsize=(20, 14))

sns.scatterplot('long','lat',hue='price',data=df_train)
plt.figure(figsize=(20, 14))

sns.scatterplot('long','lat',hue='sqft_lot',data=df_train)
plt.figure(figsize=(20, 14))

sns.scatterplot('long','lat',hue='sqft_lot15',data=df_train)
plt.figure(figsize=(20, 14))

sns.scatterplot('long','lat',hue='sqft_living',data=df_train)
plt.figure(figsize=(20, 14))

sns.scatterplot('long','lat',hue='sqft_living15',data=df_train)
data = pd.concat([df_train['price'], df_train['sqft_above']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_above', y="price", data=data, marker="+", color="g")
data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bathrooms', y="price", data=data)
data = pd.concat([df_train['price'], df_train['lat']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='lat', y="price", data=data, marker="+", color="g")
data = pd.concat([df_train['price'], df_train['long']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='long', y="price", data=data, marker="+", color="g")
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bedrooms', y="price", data=data)
df_train.bedrooms.describe()
data = pd.concat([df_train['price'], df_train['floors']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='floors', y="price", data=data)
df_train.floors.describe()
df_train.view.describe()
plt.plot(df_train.view, '+')
data = pd.concat([df_train['price'], df_train['view']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='view', y="price", data=data)
condition = df_train['condition'].value_counts()



print("Condition counting: ")

print(condition)



fig, ax = plt.subplots(ncols=2, figsize=(14,5))

sns.countplot(x='condition', data=df_train, ax=ax[0])

sns.boxplot(x='condition', y= 'price',

            data=df_train, ax=ax[1])

plt.show()
plt.figure(figsize = (12,8))

g = sns.FacetGrid(data=df_train, hue='condition',size= 5, aspect=2)

g.map(plt.scatter, "sqft_living", "price").add_legend()

plt.show()
condition = df_train['grade'].value_counts()



print("Grade counting: ")

print(condition)



fig, ax = plt.subplots(ncols=2, figsize=(14,5))

sns.countplot(x='grade', data=df_train, ax=ax[0])

sns.boxplot(x='grade', y= 'price',

            data=df_train, ax=ax[1])

plt.show()
plt.figure(figsize = (12,8))

g = sns.FacetGrid(data=df_train, hue='grade',size= 5, aspect=2)

g.map(plt.scatter, "sqft_living", "price").add_legend()

plt.show()
#Clearly view of bathrooms and bedrooms correlation



bath = ['bathrooms', 'bedrooms']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[bath[0]], df_train[bath[1]], margins=True).style.background_gradient(cmap = cm)
bath_cond = ['bathrooms', 'condition']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[bath_cond[0]], df_train[bath_cond[1]], margins=True).style.background_gradient(cmap = cm)
bed_cond = ['bedrooms', 'condition']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[bed_cond[0]], df_train[bed_cond[1]], margins=True).style.background_gradient(cmap = cm)
cond_water = ['condition', 'waterfront']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[cond_water[0]], df_train[cond_water[1]], margins=True).style.background_gradient(cmap = cm)
grade_cond = ['grade', 'condition']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[grade_cond[0]], df_train[grade_cond[1]], margins=True).style.background_gradient(cmap = cm)
grade_bed = ['grade', 'bedrooms']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[grade_bed[0]], df_train[grade_bed[1]], margins=True).style.background_gradient(cmap = cm)
grade_bath = ['grade', 'bathrooms']

cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(df_train[grade_bath[0]], df_train[grade_bath[1]], margins=True).style.background_gradient(cmap = cm)
#sqft_living vs price

sns.set(color_codes=True)

data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_living', y="price", data=data, marker="+", color="g")
df_train.loc[df_train['sqft_living'] > 13000]
#df_train = df_train.loc[df_train['id']!=8912]
#df_train[df_train['id']==8912]
#sqft_living vs price

sns.set(color_codes=True)

data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

plt.figure(figsize=(8, 6))

sns.regplot(x='sqft_living', y="price", data=data, marker="+", color="g")
df_train.loc[df_train['sqft_living'] > 11000]
df_train.loc[(df_train['grade'] == 13)]
#df_train = df_train.loc[df_train['id']!=5108]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

plt.figure(figsize=(8, 6))

sns.boxplot(x='grade', y="price", data=data)
df_train.loc[(df_train['grade'] == 3)]
df_train.loc[(df_train['price']>12) & (df_train['grade'] == 3)]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train.loc[(df_train['price']>14.5) & (df_train['grade'] == 7)]
#df_train = df_train.loc[df_train['id']!=2302]

#df_train = df_train.loc[df_train['id']!=4123]

#df_train = df_train.loc[df_train['id']!=7173]

#df_train = df_train.loc[df_train['id']!=2775]

#df_train = df_train.loc[df_train['id']!=12346]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

plt.figure(figsize=(8, 6))

sns.boxplot(x='grade', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bedrooms', y="price", data=data)
df_train.loc[df_train['bedrooms']>=10]
df_test.loc[df_test['bedrooms']>=10]
df_test.loc[df_test['id']==19745]
df_train.loc[(df_train['sqft_living']<=1620) & (df_train['sqft_living']>=1500)]
data1 = df_train.loc[(df_train['sqft_living']<=1620) & (df_train['sqft_living']>=1500)]

data2 = pd.concat([data1['sqft_living'], data1['bedrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bedrooms', y="sqft_living", data=data2)
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(data1[data1['bedrooms']==2]['sqft_living'], ax=ax)

sns.kdeplot(data1[data1['bedrooms']==3]['sqft_living'], ax=ax)

sns.kdeplot(data1[data1['bedrooms']==4]['sqft_living'], ax=ax)

sns.kdeplot(data1[data1['bedrooms']==5]['sqft_living'], ax=ax)

plt.legend(['bedrooms == 2', 'bedrooms == 3', 'bedrooms == 4', 'bedrooms == 5'])

plt.show()
# train데이터 내 bedrooms 개수와 sqft_living의 관계

data = pd.concat([df_train['sqft_living'], df_train['bedrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bedrooms', y="sqft_living", data=data)
# test데이터 내 bedrooms 개수와 sqft_living의 관계

data = pd.concat([df_test['sqft_living'], df_test['bedrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bedrooms', y="sqft_living", data=data)
df_test.loc[df_test['id']==19745].bedrooms
df_test.loc[df_test['id']==19745, 'bedrooms'] = 3

#df_test.loc[df_test['id']==19745].bedrooms = 3
df_test.loc[df_test['id']==19745].bedrooms
#train 데이터. price vs bathrooms

data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bathrooms', y="price", data=data)
#train 데이터. sqft_living vs bathrooms

data = pd.concat([df_train['sqft_living'], df_train['bathrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bathrooms', y="sqft_living", data=data)
#test 데이터. sqft_living vs bathrooms

data = pd.concat([df_test['sqft_living'], df_test['bathrooms']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='bathrooms', y="sqft_living", data=data)
df_train.loc[df_train['bathrooms']>6]
df_train.loc[(df_train['bathrooms']>=6.75) & (df_train['bathrooms']<=7.5)]
#df_train = df_train.loc[df_train['id']!=2859]

#df_train = df_train.loc[df_train['id']!=5990]
#skew_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

#skew_columns = ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']





fig, ax = plt.subplots(3, 2, figsize=(10, 15))



count = 0

for row in range(3):

    for col in range(2):

        if count == 6:

            break

        sns.kdeplot(df_train[skew_columns[count]], ax=ax[row][col])

        ax[row][col].set_title(skew_columns[count], fontsize=15)

        count+=1
#from scipy.special import boxcox1p

#lam = 0.15



#for c in skew_columns:

#    df_train[c] = boxcox1p(df_train[c], lam)

#    df_test[c] = boxcox1p(df_test[c], lam)

    

for c in skew_columns:

    df_train[c] = np.log1p(df_train[c])

    df_test[c] = np.log1p(df_test[c])
#for c in skew_columns:

#    df_train[c] = np.log1p(df_train[c].values)

#    df_test[c] = np.log1p(df_test[c].values)
fig, ax = plt.subplots(3, 2, figsize=(10, 15))



count = 0

for row in range(3):

    for col in range(2):

        if count == 6:

            break

        sns.kdeplot(df_train[skew_columns[count]], ax=ax[row][col])

        ax[row][col].set_title(skew_columns[count], fontsize=15)

        count+=1
for df in [df_train,df_test]:

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
df_train.head()
for df in [df_train,df_test]:

    df['date(new)'] = df['date'].apply(lambda x: int(x[4:8])+800 if x[:4] == '2015' else int(x[4:8])-400)

    del df['date']

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

    #df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 

    #df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built'] 

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1) #재건축 여부

    df['yr_renovated'] = df['yr_renovated'].astype('int')
df_train.head()
len(set(df_train['zipcode'].values))
data = pd.concat([df_train['price'], df_train['zipcode']], axis=1)

plt.figure(figsize=(18, 6))

sns.boxplot(x='zipcode', y="price", data=data)
## 현우님 kernel 참고 (https://www.kaggle.com/chocozzz/house-price-prediction-eda-updated-2019-03-12)

df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

# 70개 zipcode group들에 대한 mean & var 변수 추출

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

## groupby, 연산, agg. 참고 (https://datascienceschool.net/view-notebook/76dcd63bba2c4959af15bec41b197e7c/)

## reset_index 참고 (https://datascienceschool.net/view-notebook/a49bde24674a46699639c1fa9bb7e213/)

zipcode_price
print(len(df_train.columns))

print(len(df_test.columns))
df_train.columns
df_test.columns
#mean, var 변수 2개씩 추가됨

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')
print(len(df_train.columns))

print(len(df_test.columns))
# 면적 당 가격의 mean/var 이었으므로, 이를 total_size와 곱해줌

for df in [df_train,df_test]:

    df['zipcode_mean'] = df['mean'] * df['sqft_total_size']

    df['zipcode_var'] = df['var'] * df['sqft_total_size']

    #del df['mean']; del df['var']
print(len(df_train.columns))

print(len(df_test.columns))
df_train.columns
df_train.head()
#skew_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

skew_columns = ['mean', 'var', 'zipcode_mean', 'zipcode_var']

#skew_columns = ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']





fig, ax = plt.subplots(2, 2, figsize=(10, 15))



count = 0

for row in range(2):

    for col in range(2):

        if count == 4:

            break

        sns.kdeplot(df_train[skew_columns[count]], ax=ax[row][col])

        ax[row][col].set_title(skew_columns[count], fontsize=15)

        count+=1
from scipy.special import boxcox1p

lam = 0.15



for c in skew_columns:

    df_train[c] = boxcox1p(df_train[c], lam)

    df_test[c] = boxcox1p(df_test[c], lam)
#skew_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

skew_columns = ['mean', 'var', 'zipcode_mean', 'zipcode_var']

#skew_columns = ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']





fig, ax = plt.subplots(2, 2, figsize=(10, 15))



count = 0

for row in range(2):

    for col in range(2):

        if count == 4:

            break

        sns.kdeplot(df_train[skew_columns[count]], ax=ax[row][col])

        ax[row][col].set_title(skew_columns[count], fontsize=15)

        count+=1
# y: price

train_price = df_train.price.values
train_price
# df_train without price

df_train = df_train.drop('price', axis=1)
# df_train without per_price

df_train = df_train.drop('per_price', axis=1)
print(len(df_train.columns))

print(len(df_test.columns))
df_train.head()
#for df in [df_train,df_test]:

#    df = df.drop(["id", "zipcode", "long"], axis=1)
test_id = df_test.id
df_train = df_train.drop(["id", "sqft_lot15"], axis=1)

df_test = df_test.drop(["id", "sqft_lot15"], axis=1)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
# Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse = np.sqrt(-cross_val_score(model, df_train.values, train_price, scoring="neg_mean_squared_error", cv = kf))

    return (rmse)





# cross_val_score

# https://datascienceschool.net/view-notebook/266d699d748847b3a3aa7b9805b846ae/

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html



# scorer, scoring metrics

# https://scikit-learn.org/stable/modules/model_evaluation.html



# RMSLE

# https://programmers.co.kr/learn/courses/21/lessons/943

# https://www.slideshare.net/KhorSoonHin/rmsle-cost-function

# https://dacon.io/user1/41382
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



GBoost = GradientBoostingRegressor(n_estimators=8000, learning_rate=0.05, max_depth=5,

                                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10,

                                   loss='huber', random_state=4)



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=8000,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=31,

                              learning_rate=0.015, n_estimators=8000,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 1, feature_fraction = 0.9,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_child_samples = 20, reg_alpha= 0.1)



model_RF = RandomForestRegressor(max_depth = 8, 

                                n_estimators = 8000,

                                max_features = 'sqrt', 

                                n_jobs = -1)
#models = [{'model':lasso, 'name':'LASSO'}, {'model':ENet, 'name':'ENet'},

#          {'model':KRR, 'name':'KernelRidge'}, {'model':GBoost, 'name':'GradientBoosting'}, 

#          {'model':model_xgb, 'name':'XGBoost'}, {'model':model_lgb, 'name':'LightGBM'}]



models = [{'model':lasso, 'name':'LASSO'}, {'model':ENet, 'name':'ENet'},

          {'model':GBoost, 'name':'GradientBoosting'}, {'model':model_RF, 'name':'RandomForest'}, 

          {'model':model_xgb, 'name':'XGBoost'}, {'model':model_lgb, 'name':'LightGBM'}]
for m in models:

    score = rmsle_cv(m['model'])

    print("Model {} CV score : {:.4f} ({:.4f})\n".format(m['name'], score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # base_models_는 2차원 배열입니다.

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    # 각 모델들의 평균값을 사용합니다.

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(

    base_models=(ENet, GBoost, model_RF, model_xgb, model_lgb),

    meta_model=(lasso)

)



#score = rmsle_cv(stacked_averaged_models)

#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#define a rmsle evaluation function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(df_train.values, train_price)

stacked_train_pred = stacked_averaged_models.predict(df_train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(df_test.values))

print(rmsle(train_price, stacked_train_pred))
model_xgb.fit(df_train, train_price)

xgb_train_pred = model_xgb.predict(df_train)

xgb_pred = np.expm1(model_xgb.predict(df_test))

print(rmsle(train_price, xgb_train_pred))
GBoost.fit(df_train, train_price)

GBoost_train_pred  = GBoost.predict(df_train)

GBoost_pred = np.expm1(GBoost.predict(df_test))

print(rmsle(train_price, GBoost_train_pred))
model_lgb.fit(df_train, train_price)

lgb_train_pred = model_lgb.predict(df_train)

lgb_pred = np.expm1(model_lgb.predict(df_test))

print(rmsle(train_price, lgb_train_pred))
sub = pd.DataFrame()

sub['id'] = test_id

sub['price'] = stacked_pred

sub.to_csv('submission_staking.csv',index=False)
sub = pd.DataFrame()

sub['id'] = test_id

sub['price'] = xgb_pred

sub.to_csv('submission_xgb.csv',index=False)
sub = pd.DataFrame()

sub['id'] = test_id

sub['price'] = GBoost_pred

sub.to_csv('submission_GBoost.csv',index=False)
sub = pd.DataFrame()

sub['id'] = test_id

sub['price'] = lgb_pred

sub.to_csv('submission_lgb.csv',index=False)
ensemble = stacked_pred*0.4 + xgb_pred*0.2 + GBoost_pred*0.2 + lgb_pred*0.2

sub = pd.DataFrame()

sub['id'] = test_id

sub['price'] = ensemble

sub.to_csv('submission_ensemble.csv',index=False)