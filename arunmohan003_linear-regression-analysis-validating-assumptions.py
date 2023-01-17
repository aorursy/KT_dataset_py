import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

import matplotlib.style as style

from scipy import stats

from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Lasso,Ridge

import scipy

import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print(data.shape)

data.head()
data.info()
data.isnull().sum()
#helper function

#function for ploting Histogram,Q-Q plot and 

# Box plot of target and also print skewness

def target_analysis(target):

    fig = plt.figure(constrained_layout=True, figsize=(14,10))

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Histogram')

    sns.distplot(target,norm_hist=True,ax=ax1)

    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('Q-Q Plot')

    stats.probplot(target,plot=ax2)

    ax3 = fig.add_subplot(grid[:,2])

    ax3.set_title('Box Plot')

    sns.boxplot(target,orient='v',ax=ax3)

    print(f'skweness is { target.skew()}')

    plt.show()
target_analysis(data['price'])
target_analysis(np.log1p(data['price']))
# transforming logprice

data['log_price'] = np.log1p(data['price'])
data.columns
df_num = data[['sqft_living','sqft_lot','sqft_basement','sqft_above','sqft_living15','sqft_lot15','floors','grade',

             'bedrooms','bathrooms','yr_built','yr_renovated', 'condition','log_price','zipcode']]



multicoll_pairs = df_num.drop(columns=['log_price']).columns.to_list()



fig,axes = plt.subplots(7,2,figsize=(15,20))



def plot_two(feat,i,j):

    sns.boxplot(x=df_num[feat],ax=axes[i,j])

    fig.tight_layout(pad=5.0)



    



for i,feat in enumerate(multicoll_pairs):

    j = i%2 #0 or 1

    plot_two(feat,i//2,j)
# df_num = df[['sqft_living','sqft_lot','sqft_basement','sqft_above','sqft_living15','sqft_lot15','bedrooms','bathrooms','yr_built','yr_renovated','log_price']]



multicoll_pairs = df_num.drop(columns=['log_price']).columns.to_list()



fig,axes = plt.subplots(7,2,figsize=(15,20))



def plot_two(feat,i,j):

    sns.regplot(x=df_num[feat], y=df_num['log_price'], ax=axes[i,j])

    sns.scatterplot(y=df_num['log_price'],x=df_num[feat],color=('orange'),ax=axes[i,j])   

    fig.tight_layout(pad=5.0)

    



for i,feat in enumerate(multicoll_pairs):

    j = i%2 #0 or 1

    plot_two(feat,i//2,j)

data = data.drop(columns=['sqft_lot15','sqft_lot'])
data['bedrooms'].describe()
data[data['bedrooms'] == 33]
# we will drop that row

data = data[data['bedrooms'] != 33]
data['yr_sale'] = data['date'].apply(lambda x: int(str(x)[0:4]))



sns.distplot(data['yr_sale'])

plt.show()

data['yr_sale'].describe()
data['age'] = -(data['yr_built'] - data['yr_sale'])
sns.distplot(data['age'])

plt.show()

data['age'].describe()
sns.scatterplot(data['age'],data['log_price'])

plt.show()
bins = [-30,0,20,40,60,80,100,120]

labels = ['<1','0-20','20-40','40-60','60-80','80-100','>100']

data['age_binned'] = pd.cut(data['age'], bins=bins, labels=labels)



sns.countplot(data['age_binned'])

plt.show()
sns.distplot(data['yr_built'])

data['yr_built'].describe()
yr = [i for i in range(1900,2020)]

vals = [i for i in range(1,13) for j in range(10)]





dict_yr = { k:v for k,v in zip(yr,vals)}

data['yr_built_cat'] = data['yr_built'].map(dict_yr)

    
data['is_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x>0 else 0)
df = data[['sqft_living','sqft_basement','sqft_above','sqft_living15','bathrooms','bedrooms','floors','grade',

    'waterfront','view','condition','zipcode','yr_built_cat','is_renovated','age_binned','log_price']]

corr = df.corr()

# 'sqft_lot15','sqft_lot',

plt.figure(figsize=(10,10))

sns.heatmap(corr, cmap=sns.diverging_palette(20, 220, n=200))

plt.show()
df[['sqft_living','sqft_above','log_price']].corr()
df = df.drop(columns=['sqft_above'])
num_cols = ['sqft_living','sqft_basement','sqft_living15','bathrooms','bedrooms','floors','grade']

# 'sqft_lot15','sqft_lot'



X = df.drop(columns=['log_price'])

y = df['log_price']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)



X_train = X_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



feats = ['waterfront','view','condition','yr_built_cat','is_renovated','age_binned','zipcode']

one_hot_tr = pd.get_dummies(X_train[feats])

one_hot_test = pd.get_dummies(X_test[feats])

cat_train,cat_test = one_hot_tr.align(one_hot_test,join='left',axis=1)

std = StandardScaler()

std.fit(X_train[num_cols])

X_train[num_cols] = std.transform(X_train[num_cols])

X_test[num_cols] = std.transform(X_test[num_cols])
X_train = pd.concat((X_train[num_cols],cat_train),axis=1)

X_test = pd.concat((X_test[num_cols],cat_test),axis=1)
X_train.columns
reg = LinearRegression()

reg.fit(X_train,y_train)



train_pred = reg.predict(X_train)

test_pred = reg.predict(X_test)

print(f'Train mse: {np.sqrt(mean_squared_error(y_train,train_pred))}')

print(f'Test mse: {np.sqrt(mean_squared_error(y_test,test_pred))}')

print('-'*50)

print(f'Train R2: {r2_score(y_train,train_pred)}')

print(f'Test R2: {r2_score(y_test,test_pred)}')
residuals = y_train - train_pred

target_analysis(residuals)
residuals = y_train - train_pred



sns.scatterplot(y_train,residuals)

plt.show()