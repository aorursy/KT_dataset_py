import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
%matplotlib inline


from scipy import stats
from scipy.stats import norm, skew

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.shape
df.columns
plt.figure(figsize=(6,4), dpi=100)
plt.scatter(x=df['sqft_living'], y=df['price'], marker='.')
#from above plot we can see there is one extreme outlier which has highest living area but the price is very low. I can remove it for usual scenarios.
#we will check the index of the outlier row
df[df['sqft_living']>12000]
#we will drop the index 12777
df.drop(index=12777, axis=0, inplace=True)
f, axes = plt.subplots(1, 2, figsize=(20, 5), sharex=False)


plt.figure(figsize=(8,5), dpi=100)
sns.distplot(df['price'] , fit=norm, ax=axes[0]);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
axes[0].set_ylabel('Frequency')
axes[0].set_title('SalePrice distribution')

#Get also the QQ-plot
# fig = plt.figure(figsize=(8,5), dpi=100)
res = stats.probplot(df['price'], plot=axes[1])
log_price = np.log1p(df["price"])

f, axes = plt.subplots(1, 2, figsize=(20, 5), sharex=False)


plt.figure(figsize=(8,5), dpi=100)
sns.distplot(log_price , fit=norm, ax=axes[0]);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
axes[0].set_ylabel('Frequency')
axes[0].set_title('SalePrice distribution')

#Get also the QQ-plot
# fig = plt.figure(figsize=(8,5), dpi=100)
res = stats.probplot(log_price, plot=axes[1])
df['log_price']=np.log1p(df['price'])
df.head()
df_null_val=df.isnull().sum()/df.shape[0]*100.0
df_null_val
sns.heatmap(df.isnull(), yticklabels=False, cbar='viridis')
plt.subplots(figsize=(5,5), dpi=100)
sns.heatmap(df.corr(), cmap='viridis', square=True)
# for i in df.columns:
#     print(i, df[i].nunique())
col_uniqueness_df=pd.DataFrame([{'column':i, 'unique_values':df[i].nunique()} for i in df.columns])

col_uniqueness_df
from sklearn.preprocessing import LabelEncoder

for c in list(col_uniqueness_df[col_uniqueness_df['unique_values']<20]['column']):
    l=LabelEncoder()
    l.fit(list(df[c].values))
    df[c]=l.transform(list(df[c].values))
df.head()
df['total_sqft']=df['sqft_living']+df['sqft_lot']+df['sqft_above']+df['sqft_living15']+df['sqft_lot15']
plt.scatter(x=df['total_sqft'], y=df['price'], marker='.')
from itertools import combinations
from scipy.stats.stats import pearsonr


feature_list=['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15']
for i in range(2,len(feature_list)+1):
    combinations_object = combinations(feature_list, i)
    for set_obj in list(combinations_object):
        added_vector=np.zeros(df.shape[0])
        for col in set_obj:
            list_vector=np.array(list(df[col]))
            added=np.add(added_vector,list_vector)
            added_vector=added
        if pearsonr(added_vector, np.array(df['log_price']))[0]>0.6:
            print (set_obj)
            print (pearsonr(added_vector, np.array(df['log_price'])))
            
df['new_sqft_feature']=df['sqft_living']+df['sqft_above']+df['sqft_living15']
numeric_feats = df.dtypes[df.dtypes != "object"].index
# numeric_feats


# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head
new_df=df.copy()
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    if feat not in ['price','log_price']:
        new_df[feat] = boxcox1p(new_df[feat], lam)
numeric_feats = df.dtypes[df.dtypes != "object"].index
# numeric_feats


# Check the skew of all numerical features
skewed_feats = new_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head
new_df.head()
all_dummy=pd.get_dummies(new_df.drop(columns=['id','date'], axis=1))
all_dummy.head()
new_df.dtypes
import xgboost
reg=xgboost.XGBRegressor()
reg
n_estimators=[200,500,1000,1200, 1300, 1400, 1500]
max_depth=[2,3,5,10,15]
booster=['gbtree','gblinear']
learning_rate=[0.05, 0.1, 0.15, 0.2]
min_child_weight=[1,2,3,4,5]

hyperparameter_grid={
    'n_estimators':n_estimators  ,
    'max_depth':max_depth  ,
    'booster':booster  ,
    'learning_rate': learning_rate  ,
    'min_child_weight':   min_child_weight
}
# new_df.groupby('yr_built').count()
new_df.head(2)
# df.groupby('zipcode').count()
from sklearn.model_selection import train_test_split

X=new_df[list( set(new_df.columns) - set(['id','date','price', 'lat', 'long', 'log_price', 'total_sqft', 'waterfront', 'view', ]) )]
Y=new_df['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

pred_ds=[]
for i in range(len(y_pred)):
    pred_ds.append({'y_actual':list(y_test)[i], 'y_predicted':list(y_pred)[i]})

pred_df=pd.DataFrame(pred_ds)
pred_df['actual_price']=pred_df['y_actual'].apply(lambda x: np.expm1(x))
pred_df['predicted_price']=pred_df['y_predicted'].apply(lambda x: float(np.expm1(x)))
pred_df.head()
from sklearn.metrics import mean_squared_error
from math import sqrt

print (sqrt(mean_squared_error(y_test, y_pred, squared=False)))
from sklearn.model_selection import RandomizedSearchCV


randomcv=RandomizedSearchCV(
            estimator=reg,
            param_distributions=hyperparameter_grid,
            cv=10, n_iter=50,
            scoring='neg_mean_absolute_error', n_jobs=8,
            verbose=5,
            return_train_score=True,
            random_state=42
)
# randomcv.fit(X,Y)
best_xgb=randomcv.best_estimator_
# best_xgb
# randomcv.__dict__
print(randomcv.best_score_)

y_pred=best_xgb.predict(X_test)

best_xgb_pred_ds=[]
for i in range(len(y_pred)):
    best_xgb_pred_ds.append({'y_actual':list(y_test)[i], 'y_predicted':list(y_pred)[i]})

best_xgb_pred_df=pd.DataFrame(best_xgb_pred_ds)
best_xgb_pred_df['actual_price']=best_xgb_pred_df['y_actual'].apply(lambda x: np.expm1(x))
best_xgb_pred_df['predicted_price']=best_xgb_pred_df['y_predicted'].apply(lambda x: float(np.expm1(x)))
best_xgb_pred_df.head()
print (sqrt(mean_squared_error(y_test, y_pred, squared=False)))
plt.figure(figsize=(10,10), dpi=100)
plt.scatter(best_xgb_pred_df['actual_price'], best_xgb_pred_df['predicted_price'], marker='.')
plt.figure(figsize=(10,10), dpi=100)
plt.scatter(best_xgb_pred_df['actual_price'], best_xgb_pred_df['predicted_price'], marker='.')



import xgboost
reg=xgboost.XGBRegressor()
df.head(2)
df.columns
x_cols=['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
X=df[x_cols]
Y=df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.feature_selection import SelectKBest, f_regression
# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
...
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
X_train_fs
X_train_fs.shape
reg.fit(X_train_fs, y_train)
pred=reg.predict(X_test_fs)
type(y_test)
pred_list=[]
y_test_list=list(y_test)
for i in range(len(pred)):
    pred_list.append({'prediction':pred[i], 'actual':y_test_list[i], 'diff':pred[i]-y_test_list[i]})
pred_df=pd.DataFrame(pred_list)

pred_df
from sklearn.metrics import mean_squared_error
from math import sqrt

print (sqrt(mean_squared_error(y_test, pred, squared=False)))
sns.heatmap(X.corr())
X.corr()
df.corr()
corr=df.corr()
corr[corr]
# high_tgt_corr_cols=corr['price']
# for r, idx in corr[corr['price']>0.35].iterrows():
#     print (idx)

corr[corr['price']>0.35].index
high_corr_cols=list(corr[corr['price']>0.35].index)

high_corr_cols.remove('price')

X=df[high_corr_cols]
Y=df['price']
list(corr[corr['price']>0.35].index).remove('price')
X.head()
reg_2=xgboost.XGBRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

reg_2.fit(X_train, y_train)
pred=reg_2.predict(X_test)

pred_list=[]
y_test_list=list(y_test)
for i in range(len(pred)):
    pred_list.append({'prediction':pred[i], 'actual':y_test_list[i], 'diff':pred[i]-y_test_list[i]})
pred_df=pd.DataFrame(pred_list)

pred_df
from sklearn.model_selection import RandomizedSearchCV


print (sqrt(mean_squared_error(y_test, pred, squared=False)))
X.head()
df.shape
from sklearn.feature_selection import SelectKBest, f_regression
# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
# X_test_fs = fs.transform(X_test)
X_train.shape
X_train_fs.shape
