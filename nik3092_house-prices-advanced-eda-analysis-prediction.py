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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.pandas.set_option('display.max_columns',None)



from scipy import stats

from scipy.stats import norm,skew

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor





# for warnings

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# Remove unnecessary column from the dataset.



df_train.drop('Id',axis=1,inplace=True)

df_train.head()
df_test.drop('Id',axis=1,inplace=True)

df_test.head()
print("Train Shape: ",df_train.shape)

print("Test Shape: ",df_test.shape)
missing_percentage = (df_train.isnull().sum()/len(df_train))*100

missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending = False)

missing_percentage
df_train.drop(['PoolQC','MiscFeature','Alley','Fence','3SsnPorch'],axis=1,inplace=True)
missing_percentage_test = (df_test.isnull().sum()/len(df_test))*100

missing_percentage_test = missing_percentage_test[missing_percentage_test > 0].sort_values(ascending = False)

missing_percentage_test
df_test.drop(['PoolQC','MiscFeature','Alley','Fence','3SsnPorch'],axis=1,inplace=True)
df_train.skew()
df_test.skew()
plt.figure(figsize = (15,5))

plt.subplot(1,2,1)

fig1 = sns.distplot(df_train['SalePrice'],color = 'b',fit=norm)

(mu, sigma) = norm.fit(df_train['SalePrice'])

plt.title("Before Transformation",color = 'r',size=15)

plt.xlabel("SalePrice",size=15,color='r')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])



plt.subplot(1,2,2)

target = np.log(df_train['SalePrice'])

fig2 = sns.distplot(target,color = 'b',fit=norm,label = "After Transformation")

(mu, sigma) = norm.fit(target)

plt.title("After Transformation",color = 'r',size=15)

plt.xlabel("SalePrice",size=15,color='r')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])



plt.tight_layout()

plt.show()
num_feat = list(col for col in df_train.columns if df_train[col].dtypes != 'object')

print('Numerical Features are: {}'.format(len(num_feat)))



cat_feat = list(col for col in df_train.columns if df_train[col].dtypes == 'object')

print('Categorical Features are: {}'.format(len(cat_feat)))
plt.figure(figsize=[20,15])

corr = df_train[num_feat].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr,annot=True,cmap='coolwarm',mask = mask,linewidths=0.4,fmt='.1f',vmin=-1,vmax=1,center=0,

            cbar_kws={'orientation':'horizontal'})

plt.show()
df_train[num_feat].corr()['SalePrice'].sort_values(ascending=False).nlargest(20)
plt.figure(figsize=[15,8])

plt.subplot(2,1,1)

fig = sns.countplot(df_train['MoSold'])

plt.title("\nIn Which Month House Selling High?\n",color='r',size=25)

plt.xlabel("\nMonth",color='magenta',size=15)

plt.subplot(2,1,2)

fig1 = sns.countplot(df_train['YrSold'])

plt.title("\nIn Which Year house Selling High?\n",color='r',size=25)

plt.xlabel("\nYear",color='magenta',size=15)

plt.tight_layout()

plt.show()
plt.figure(figsize = [15,8])

sns.lineplot(df_train['YrSold'],df_train['SalePrice'],color='tomato')

plt.title("\nFluctuation of House Price over the Years",size = 20,color = 'red')

plt.xlabel('\nYr Selling',color='blue',size=15)

plt.ylabel('Selling Price',color='blue',size=15)

plt.show()
plt.figure(figsize=[15,12])

grp_order = df_train.groupby(['Neighborhood'])['SalePrice'].mean().sort_values(ascending=False).index

sns.barplot(y = df_train['Neighborhood'], x = df_train['SalePrice'],order = grp_order,ci=None,orient='h')

plt.title("\nWhich Neighborhood Is Popular?\n",size=25,color='r')

plt.xlabel("\nHouse Selling Price",color='magenta',size=15)

plt.ylabel("Neighborhood",color = 'magenta',size=25)

plt.yticks(fontsize=15)

plt.show()
df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)

df_train['OverallCond'] = df_train['OverallCond'].astype(str)

df_train['YrSold'] = df_train['YrSold'].astype(str)

df_train['MoSold'] = df_train['MoSold'].astype(str)
cat_feat = list(col for col in df_train.columns if df_train[col].dtypes == 'O')

print('Categorical Features now becomes: {}'.format(len(cat_feat)))
# Plotting for the Category Features



f = pd.melt(df_train, value_vars = sorted(cat_feat))

g = sns.FacetGrid(f,col = 'variable',col_wrap=4,sharex=False,sharey=False)

plt.xticks(rotation = 90)

g = g.map(sns.countplot,'value') 

[plt.setp(ax.get_xticklabels(),rotation = 90) for ax in g.axes.flat]

g.fig.tight_layout()

plt.show()
# checking outliers for the Numerical Features



fig,ax=plt.subplots(18,2,figsize=(15,60))

def graph(x,y,r,c,title):

    sns.scatterplot(df_train[num_feat][x],y,ax=ax[r][c])

    ax[r][c].set_xlabel(x)

    fig.tight_layout(pad=5.0)



for r,col in enumerate(num_feat):

    c=r%2

    graph(col,df_train['SalePrice'],r//2,c,col)
def outlier(df_train):

    for col in df_train.columns:

        if (((df_train[col].dtype)=='float64') | ((df_train[col].dtype)=='int64')):

            percentiles = df_train[col].quantile([0.25,0.75]).values

            df_train[col][df_train[col] <= percentiles[0]] = percentiles[0]

            df_train[col][df_train[col] >= percentiles[1]] = percentiles[1]

        else:

            df_train[col]=df_train[col]

    return df_train



df_train = outlier(df_train)
for col in df_train.columns:

    if df_train[col].isnull().sum() > 50:

        df_train[col].fillna(0,inplace=True)

    elif df_train[col].dtypes == 'O':

        df_train[col].fillna(df_train[col].mode()[0],inplace=True)

    else:

        df_train[col].fillna(df_train[col].mean(),inplace = True)
df_train.isnull().values.any()
for col in df_test.columns:

    if df_test[col].isnull().sum() > 50:

        df_test[col].fillna(0,inplace=True)

    elif df_test[col].dtypes == 'O':

        df_test[col].fillna(df_test[col].mode()[0],inplace=True)

    else:

        df_test[col].fillna(df_test[col].mean(),inplace = True)
df_test.isnull().values.any()
encoder = LabelEncoder()



for col in cat_feat:

    df_train[col] = encoder.fit_transform(df_train[col].astype(str))
print(df_train.shape)

df_train.head()
cat_feat_test = list(col for col in df_test.columns if df_test[col].dtypes == 'object')

print("Category Features of Test data are: ",len(cat_feat_test))



for col in cat_feat_test:

    df_test[col] = encoder.fit_transform(df_test[col].astype(str))
print(df_test.shape)

df_test.head()
X = df_train.drop('SalePrice',axis=1)

y = df_train['SalePrice']
print(X.shape)

X.head()
y.head().to_frame()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)



print("X_train Shape:",X_train.shape)

print("X_test Shape:",X_test.shape)

print("y_train Shape:",y_train.shape)

print("y_test Shape:",y_test.shape)
plt.figure(figsize = [15,25])

from sklearn.ensemble import ExtraTreesRegressor

extra_reg = ExtraTreesRegressor()

extra_reg.fit(X_train,y_train)



features = pd.Series(extra_reg.feature_importances_,index= X_train.columns).plot(kind = 'barh')

plt.title("\nBest Features",size = 20,color = 'crimson')

plt.ylabel('Features',size=20,color='purple')

plt.yticks(fontsize = 10)

plt.show()
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.0385, 

                                       n_estimators=3500,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose= 0,

                                       )
lightgbm.fit(X_train,y_train)
lightgbm.score(X_test,y_test)
prediction = lightgbm.predict(df_test)

print(prediction)
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
final_output = pd.DataFrame({'Id':submission['Id'],'SalePrice' : prediction})

final_output.to_csv('Submission_lightgbm.csv',index = False)

final_output.head()