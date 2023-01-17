!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887
!apt update && apt install -y libsm6 libxext6
%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import train_cats, proc_df, rf_feat_importance
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn import metrics
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
# !ls ../input
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)
df_train.head()

fig ,ax = plt.subplots()
ax.scatter(x=df_train['GrLivArea'],y = df_train['SalePrice'])
plt.title('Showing Outliers')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()
#Deleting outliers
df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 200000)].index, inplace=True)
# check the graph again
fig ,ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

df_train['SalePrice'].head()
df_train["SalePrice"] = np.log(df_train["SalePrice"])

sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
# from scipy.special import boxcox1p
# df_train['SalePrice'] = boxcox1p(df_train['SalePrice'], 0.15)
# #df_train["SalePrice"] = np.log(df_train["SalePrice"])

# sns.distplot(df_train['SalePrice'] , fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(df_train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# # #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
# plt.show()

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()
n_train = df_train.shape[0]
n_test = df_test.shape[0]
y = df_train.SalePrice.values
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop('SalePrice', axis=1, inplace=True)
print("Size of all data : {}".format(all_data.shape))
all_data.head()
train_cats(all_data)
all_data,Alley,nas=proc_df(all_data,'Alley')
all_data.head()
all_data.columns
dd=['BsmtFinSF1_na', 'BsmtFinSF2_na', 'BsmtFullBath_na', 'BsmtHalfBath_na', 'BsmtUnfSF_na', 'GarageArea_na',
       'GarageCars_na', 'GarageYrBlt_na', 'LotFrontage_na', 'MasVnrArea_na', 'TotalBsmtSF_na']
all_data.drop(dd, axis=1, inplace=True)

all_data.shape
all_data.head()
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
train = all_data[:n_train]
test = all_data[n_train:]
train.head()
test.head()
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid),y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 88
n_trn = len(train) - n_valid
X_train, X_valid = split_vals(train, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestRegressor(n_jobs=-1, random_state=1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=2, n_estimators=46, oob_score=True, max_features=0.5, 
                          max_depth=10)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=1, n_estimators=25, oob_score=True, max_features=0.6)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=1, n_estimators=40, oob_score=True, max_features=0.6, 
                          )
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=6, n_estimators=60, oob_score=True, max_features=0.5, 
                          max_depth=16, min_samples_leaf=3)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=6, n_estimators=60, oob_score=True, max_features=0.4, 
                          max_depth=16, min_samples_leaf=3, max_leaf_nodes=450)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=100, oob_score=True, max_features=0.5, 
                          max_depth=14, min_samples_leaf=3, max_leaf_nodes=400, min_impurity_decrease=0.00001)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=160, oob_score=True, max_features=0.5, 
                          max_depth=14, min_samples_leaf=2, max_leaf_nodes=400, min_impurity_decrease=0.00001)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1, random_state=10, n_estimators=160, oob_score=True, max_features=0.5, 
                          max_depth=None, min_samples_leaf=2, max_leaf_nodes=250, min_impurity_decrease=0.00001,
                          min_impurity_split=None)

m.fit(X_train,y_train)
print_score(m)
SalePrice = m.predict(test)
df_sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_sample.head()
SalePrice = np.exp(SalePrice)
df_sample['SalePrice'] = SalePrice
df_sample.head()
df_sample.to_csv('Home_price.csv', columns=['Id','SalePrice'], index=False)
df_sample.SalePrice.head()