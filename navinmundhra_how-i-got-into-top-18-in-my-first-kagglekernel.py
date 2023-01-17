import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns #data visualization

from scipy import stats

from scipy.stats import norm

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing as prep

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



print('train data shape: ', train.shape, '\ntest data shape: ', test.shape)

print('Reading done!')
train.drop('Id', axis = 1, inplace = True)

test.drop('Id', axis = 1, inplace = True)

print("Dropped redundant column 'Id' from both train and test data")



target = 'SalePrice'

print('Target variable saved in a variable for further use!')
def getnumcatfeat(df):

    

    """Returns two lists of numeric and categorical features"""

    

    numfeat, catfeat = list(df.select_dtypes(include=np.number)), list(df.select_dtypes(exclude=np.number))

    return numfeat, catfeat



numfeat, catfeat = getnumcatfeat(train)

numfeat.remove(target)



print('Categorical & Numeric features seperated in two lists!')
fig, a = plt.subplots(nrows=1, ncols=2, figsize = (20,7))





sns.distplot(train[target], fit = norm, ax=a[0])

(mu, sig) = norm.fit(train[target]) 

a[0].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,train[target].skew(),train[target].kurt())])

a[0].set_ylabel('Frequency')

a[0].axvline(train[target].mean(), color = 'Red') 

a[0].axvline(train[target].median(), color = 'Green') 

a[0].set_title(target + ' distribution')



temp=np.log1p(train[target])



sns.distplot(temp, fit = norm, ax=a[1])

(mu, sig) = norm.fit(temp) 

a[1].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,temp.skew(),temp.kurt())])

a[1].set_ylabel('Frequency')

a[1].axvline(temp.mean(), color = 'Red')

a[1].axvline(temp.median(), color = 'Green') 

a[1].set_title('Transformed '+ target + ' distribution')



train[target] = np.log1p(train[target])



plt.show()
temp = 'OverallQual'

f,a = plt.subplots(figsize=(8,6))

sns.boxplot(x= temp, y = target, data = train)

plt.show()
train.drop(train[((train[temp]==3) | (train[temp]==4)) & (train[target]<10.75)].index, inplace=True)
temp = 'GrLivArea'

f,a = plt.subplots(figsize=(8,6))

sns.scatterplot(x=temp, y=target, data=train)

corr, _ = stats.pearsonr(train[temp], train[target])

plt.title('Pearsons correlation: %.3f' % corr)

plt.show()
train.drop(train[(train[temp]>4000) & (train[target]<12.5)].index, inplace=True)
temp = 'GarageCars'

f,a = plt.subplots(figsize=(8,6))

sns.boxplot(x=temp,y=target,data=train)

plt.show()
temp = 'TotalBsmtSF'

f,a = plt.subplots(figsize=(8,6))

sns.scatterplot(x=temp,y=target,data=train)

corr, _ = stats.pearsonr(train[temp], train[target])

plt.title('Pearsons correlation: %.3f' % corr)

plt.show()
cor = train.corr()

f,a = plt.subplots(figsize=(15,10))

sns.heatmap(cor)

plt.show()
topn = 20

print('Top ', topn, ' correlated features to target features')

cor[target].sort_values(ascending=False)[1:(topn+1)]
s = cor.unstack().sort_values(ascending = False)[len(cor):]



topn = 20

print('Top', int(topn/2), 'correlated features\n')

s[:topn:2]

train_labels = train[target].reset_index(drop=True)

train_features = train.drop(target, axis=1)

test_features = test



df = pd.concat([train_features, test_features]).reset_index(drop=True)



## dropping the columns with multicollinearity

df.drop(['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'], axis=1, inplace=True)



df.shape
# topn = 20

# # print(cor[target].sort_values()[:topn])

# temp = list(cor[target].sort_values()[:topn].index)

# df.drop(temp, axis=1, inplace=True)



# print('Removed least ', topn, ' correlated features')
numfeat, catfeat = getnumcatfeat(df)
temp = 0
assert(temp<len(catfeat))

print('Feature name: ', catfeat[temp], '\nUnique values: ', df[catfeat[temp]].unique(), 

      '\nData type: ', df[catfeat[temp]].dtype, '\nValue ', temp, ' out of ', len(catfeat))

temp +=1
temp = 0
assert(temp<len(numfeat))

print('Feature name: ', numfeat[temp], '\nUnique values: ', df[numfeat[temp]].unique(), 

      '\nData type: ', df[numfeat[temp]].dtype, '\nIndex ', temp, ' out of ', len(numfeat)-1)

temp +=1
df['YearBuilt'] = df['YearBuilt'].astype('category')

df['YearRemodAdd'] = df['YearRemodAdd'].astype('category')

df['MoSold'] = df['MoSold'].astype('category')

df['YrSold'] = df['YrSold'].astype('category')
numfeat, catfeat = getnumcatfeat(df)
temp = df.isnull().sum().sort_values(ascending=False)/df.shape[0]*100

temp = temp[temp>0]

temp = pd.DataFrame(temp, columns = ['MissPercent'])



f,a = plt.subplots(figsize=(12,10))



sub = sns.barplot(x='MissPercent', y = temp.index, data=temp, orient='h')

plt.title('Percent of missing values, size = '+ str(temp.shape[0]))

## Annotating the bar chart

for p,t in zip(sub.patches, temp['MissPercent']):

    plt.text(2.3+p.get_width(), p.get_y()+p.get_height()/2, '{:.2f}'.format(t), ha='center', va = 'center')



sns.despine(top=True, right=True)



plt.show()
def findit(df, strin):

    """

    CONVENIENCE FUNCTION FOR ANOTHER FUNCTION

    """



    temp = []

    for col in df.columns:

        if col[:len(strin)]==strin:

            temp.append(col)

    if len(temp)==0:

        return 0

    return temp



def fillit(df, strin):

    """

    

    CONVENIENCE FUNCTION

    

    Finds features beginning with 'strin' in its beginning.

    Then fills null values of categorical and numeric features

    with str('None') and int(0) values respectively.

    

    """

    temp = findit(df,strin)

    for col in temp:

        if df[col].dtype == object:

            df[col].fillna('None', inplace=True)

        else:

            df[col].fillna(0, inplace=True)

    return None
df['PoolQC'].fillna('None', inplace=True)

df['MiscFeature'].fillna('None', inplace=True)

df['Alley'].fillna(df['Alley'].mode()[0], inplace=True)

df['Fence'].fillna('None', inplace=True)

df['FireplaceQu'].fillna("None", inplace=True)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

fillit(df,'Garage')

fillit(df,'Bsmt')

fillit(df,'Mas')

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)

df['Functional'].fillna('Typ', inplace=True)

# Replace the missing values in each of the columns below with their mode

df['Electrical'].fillna(df['Electrical'].mode()[0],inplace=True)

df['KitchenQual'].fillna(df['KitchenQual'].mode()[0],inplace=True)

df['Exterior1st'].fillna(df['Exterior1st'].mode()[0],inplace=True)

df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0],inplace=True)

df['SaleType'].fillna(df['SaleType'].mode()[0],inplace=True)

df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)

df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mode()[0],inplace=True)

df.isnull().sum().sort_values(ascending=False)
temp = list(df[df['PoolQC']=='None']['PoolArea'].unique())

df['PoolArea'] = df['PoolArea'].replace(temp,0)
def minfun(lamb):

    return round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb)).skew(), 2)



def retlamb(df, numfeat, tol, n_iter=100):

    """

    

    CONVENIENCE FUNCTION

    

    Returns optimized values of lambda to be used in

    boxcox transformation for each feature so that 

    skewness is minimized

    

    """

    valLambda = {}

    lim1, lim2 = 0, 4

    idx=0

    for temp in numfeat:

        lim1, lim2 = 0, 2

        for i in range(n_iter):

            lamb1=0.5*(lim1+lim2)-tol

            cal1 = round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb1)).skew(), 4)

            lamb2=0.5*(lim1+lim2)+tol

            cal2 = round(pd.Series(stats.boxcox(1+df[temp],lmbda=lamb2)).skew(), 4)

            if abs(cal1)<abs(cal2):

                lim2=lamb2

            elif abs(cal1)>=abs(cal2) :

                lim1=lamb1

        valLambda[idx] = 0.5*(lim1+lim2)

        idx+=1

    return valLambda
valLambda = retlamb(df,numfeat, tol=0.0001, n_iter=1000)

valLambda
temp = 2

lamb = valLambda[temp]

# temp, lamb = 0, 1

f,a = plt.subplots(nrows=1, ncols=2, figsize=(16,6))



sns.distplot(df[numfeat[temp]], ax=a[0], kde=False)

a[0].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,df[numfeat[temp]].skew(),df[numfeat[temp]].kurt())])

a[0].set_title('Distribution of '+ numfeat[temp])



tempdf = pd.Series(stats.boxcox(1+df[numfeat[temp]],lmbda=lamb))



sns.distplot(tempdf, ax=a[1], kde=False)

a[1].legend(['$\mu=$ {:.2f}, $\sigma=$ {:.2f}, Skew = {:.4f}, Kurtosis = {:.4f}'.format(mu,sig,tempdf.skew(),tempdf.kurt())])

a[1].set_title('Transformed distribution of '+ numfeat[temp])



valLambda[temp] = lamb



plt.show()
for temp, lamb in valLambda.items():

    df[numfeat[temp]] = stats.boxcox(1+df[numfeat[temp]],lmbda=lamb)

    

# ## SIMPLE SIMPLE FIX FOR ALL DRAMA DONE IS ALL THE ABOVE 3 CELLS but my approach gave better results so I happily stuck with it

# for temp in range(len(numfeat)):

#     df[numfeat[temp]] = stats.boxcox(1+df[numfeat[temp]], stats.boxcox_normmax(df[numfeat[temp]] + 1))
df[numfeat].skew()
df = pd.get_dummies(df).reset_index(drop=True)

df.shape
minmaxscalar = prep.MinMaxScaler()

df1 = pd.DataFrame(minmaxscalar.fit_transform(df), columns = df.columns)

df1.head()
X = df1.iloc[:len(train_labels),:]

X_test = df1.iloc[len(train_labels):, :]

y = train_labels



X.shape, X_test.shape, y.shape
from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV 

from sklearn.linear_model import Lasso



# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



kf = KFold(n_splits = 7, random_state=0, shuffle=True)



scores = {}
def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.009, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)
score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgb'] = (score.mean(), score.std())
xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:squarederror',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)
score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgb'] = (score.mean(), score.std())
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 50)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 300, 100)]



# Minimum number of samples required to split a node

min_samples_split = [int(x) for x in np.linspace(2,500,100)]



# Minimum number of samples required at each leaf node

min_samples_leaf = [int(x) for x in np.linspace(2,500,100)]



# Create the random grid

rforestgrid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}
rforest = RandomForestRegressor()
# clf = RandomizedSearchCV(rforest, rforestgrid, cv=5, n_iter=50, n_jobs=1)

# # search = clf.fit(X,y)
# search.best_params_
rforest_tuned = RandomForestRegressor(n_estimators = 1000,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       max_features = 'auto',

                                       max_depth = 100

                                      )
score = cv_rmse(rforest_tuned)

print("rforest: {:4f} ({:.4f})".format(score.mean(), score.std()))

scores['rforest'] = (score.mean(), score.std())
lasso = Lasso(alpha=0.000328)
score = cv_rmse(lasso)

print("lasso: {:4f} ({:.4f})".format(score.mean(), score.std()))

scores['lasso'] = (score.mean(), score.std())
model = lasso.fit(X, y)
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.shape
model.predict(X_test).shape
submission.iloc[:,1] = np.floor(np.expm1(model.predict(X_test)))
submission.to_csv("submission_try4.csv", index=False)
## Downloading the submission file



from IPython.display import FileLink

FileLink('submission_try4.csv')