# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_train =  pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Any results you write to the current directory are saved as output.
df_train.columns
df_train['SalePrice'].describe()
plt.hist(df_train['SalePrice'], 50)
sns.distplot(df_train['SalePrice']);
print("skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', alpha=.2)



gb = df_train.groupby('Neighborhood')



fig, ax = plt.subplots()

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in gb:

    ax.plot(group[var], group['SalePrice'], marker='o', linestyle='', ms=5, label=name, alpha=0.5)

ax.legend()
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var], df_train['Neighborhood']], axis=1)

# data.plot.scatter(x=var, y='SalePrice', alpha=.2, c='Neighborhood')



cat = 'OverallQual'

gb = df_train.groupby(cat)



fig, ax = plt.subplots()

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in gb:

    ax.plot(group[var], group['SalePrice'], marker='o', linestyle='', ms=5, label=name, alpha=0.2)

ax.legend()
corrmat = df_train.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat, square = True, vmin=0.5)
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm,cbar=True, annot = True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',

      'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size=2, plot_kws=dict(alpha=0.1))

plt.show()
def checkNull(df):

    totalNull = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing = pd.concat([totalNull, percent], axis=1, keys=['Total', 'Percent'])

    print(missing.head(20))

    return missing



missing_data = checkNull(df_train)

_ = checkNull(df_test)
#delete missing data

df_train = df_train.drop(missing_data[missing_data['Total']>1].index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_test = df_test.drop(missing_data[missing_data['Total']>1].index,1)

df_train.isnull().sum().max()

_ = checkNull(df_test)
# display columns and data with NA

lc = ('BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath',

      'GarageCars','GarageArea', 'BsmtFinSF1')

r=pd.Series()



for c in lc:

    try:

        tr = df_test[c].isnull()

    except KeyError:

        print(c)

        pass

    r = r | tr

    

    

print(df_test.ix[r,lc])



def cat_imputation(df, column, value):

    df.loc[df[column].isnull(), column] = value

    

for c in lc:

    cat_imputation(df_test, c, 0.0)
#outlier

#sale price univariate analysis

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

print(type(saleprice_scaled))

sortedIndex = saleprice_scaled[:,0].argsort()

low_range = saleprice_scaled[sortedIndex][:10]

high_range = saleprice_scaled[sortedIndex][-10:]



print("low range",low_range)

print("high range",high_range)
#

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', alpha=.2)



# gb = df_train.groupby('Neighborhood')



#fig, ax = plt.subplots()

#ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#for name, group in gb:

#    ax.plot(group[var], group['SalePrice'], marker='o', linestyle='', ms=5, label=name, alpha=0.5)

#ax.legend()
print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])

df_train=df_train.drop(df_train[df_train['Id'].isin([1299, 524])].index)
#bivariate analysis saleprice/TotalBsmtSF

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', alpha=.2)
sns.distplot(df_train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm)

fig=plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
# ok. Checking GrLivArea

sns.distplot(df_train['GrLivArea'], fit=norm)

fig=plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

sns.distplot(df_train['GrLivArea'], fit=norm)

fig=plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# insert column, HasBsmt.

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

# setup boolean.

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt']=1
#transform TotalBsmtSF

df_train.loc[df_train['HasBsmt']==1, 'TotalBsmtSF']=np.log(df_train['TotalBsmtSF'])

# len(df_train.loc[df_train['HasBsmt']==1, 'TotalBsmtSF'])

# len(np.log(df_train['TotalBsmtSF']))

#len(df_train)
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'], alpha=0.2);
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], 

           df_train[df_train['TotalBsmtSF']>0]['SalePrice'], alpha=0.2)
# check data type

for col in df_test.columns:

    t1 = df_test[col].dtype

    t2 = df_train[col].dtype

    if t1 != t2:

        print(col, t1, t2)



 

# convert to type of int64

c = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for cols in c:

    tmp_col = df_train[cols].astype(pd.np.float64)

    tmp_col = pd.DataFrame({cols: tmp_col})

    del df_train[cols]

    df_train = pd.concat((df_train, tmp_col), axis=1)
def oneHot(df):

    for c in df.columns:

        if df[c].dtype == np.object:

            df = pd.concat((df, pd.get_dummies(df[c], prefix=c)), axis=1)

            del df[c]        

    return df



df_train = oneHot(df_train)

df_test = oneHot(df_test)

#realign features

df_test['SalePrice']=-1



def featuresReAlignedTo(dfSource, dfTarget):

    col_target = dfTarget.columns

    col_src = dfSource.columns

    for col in col_target:

        if col in col_src:

            pass

        else:

            del dfTarget[col]

    return dfTarget



df_test = featuresReAlignedTo(df_train, df_test)

df_train = featuresReAlignedTo(df_test, df_train)   



print(len(df_test.columns), len(df_train.columns))

#207 207         
# RF regression 

from sklearn.ensemble import RandomForestRegressor



etr = RandomForestRegressor(n_estimators=400)

train_y = df_train['SalePrice']

train_x = df_train.drop(['SalePrice','Id'], axis=1)

etr.fit(train_x, train_y)

imp = etr.feature_importances_

imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})

print(imp.sort_values(by='score', ascending=False)[:20])
# GB regression 

from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor(

                n_estimators=400,

                max_features='sqrt', 

                )



gbr.fit(train_x, train_y)

gbimp = pd.DataFrame({'feature': train_x.columns, 'score': gbr.feature_importances_})

print(gbimp.sort_values(by='score', ascending=False)[:20])
# use pipeline, standardscaler, and cross validation to benchmark, training accuracy around 91.8%

from sklearn.pipeline import make_pipeline

from sklearn.cross_validation import cross_val_predict

from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn import metrics



clf = make_pipeline(StandardScaler(), gbr)

scores = cross_val_score(clf, train_x, train_y, cv=10)

print(scores, np.mean(scores))

#[ 0.90837653  0.93912429  0.93398808  0.90203234  0.89475431  0.91904469

#  0.91248096  0.93576603  0.93703954  0.89759561] 0.918020236656
# Generate output.

clf.fit(train_x, train_y)

result = clf.predict(df_test.drop(['SalePrice','Id'], axis=1))

print(result[:10])

pred= np.exp(result)

print(pred[:10])

df_submission = pd.concat([df_test['Id'],pd.Series(pred, name='SalePrice')],axis=1)

df_submission.to_csv('gbr-cv-std-2.csv', index=None)