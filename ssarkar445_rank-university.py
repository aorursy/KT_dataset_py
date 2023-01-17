import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



import sklearn as sk

from sklearn.metrics import r2_score



from sklearn.model_selection import train_test_split
world_rank = pd.read_csv('../input/cwurData.csv')
world_rank.info()

world_rank.describe([0.25,0.50,0.75,0.99])

world_rank.columns

world_rank[list(world_rank.dtypes[world_rank.dtypes=='object'].index)].head()
world_rank.shape
world_rank.isnull().all()

world_rank.isnull().any()

round(100*(world_rank.isnull().sum()/len(world_rank.index)),2)
world_rank= world_rank[~world_rank.broad_impact.isnull()]
round(100*(world_rank.isnull().sum()/len(world_rank.index)),2)
world_rank.dtypes[world_rank.dtypes!='object']
world_rank = world_rank.drop('year',axis='columns')
sns.pairplot(world_rank,x_vars = ['national_rank','quality_of_education','alumni_employment',

                       'quality_of_faculty','publications'],y_vars = 'world_rank')

sns.pairplot(world_rank,x_vars = ['influence','citations','broad_impact',

                       'patents','score'],y_vars = 'world_rank')

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(world_rank.corr(),annot = True,cmap='summer')
sns.lmplot(x = 'broad_impact',y= 'world_rank',data = world_rank)

plt.title('world rank vs broad impact')

plt.show()
sns.lmplot(x = 'influence',y= 'world_rank',data = world_rank)

plt.title('world rank vs broad impact')

plt.show()
country_frame = pd.get_dummies(world_rank.country,drop_first=True)
df_world_rank = pd.concat([world_rank,country_frame],axis = 'columns')
df_world_rank = df_world_rank.drop(list(df_world_rank.dtypes[df_world_rank.dtypes=='object'].index),axis='columns')
df_train,df_test = train_test_split(df_world_rank,train_size = 0.7,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
vars = ['world_rank', 'national_rank', 'quality_of_education',

       'alumni_employment', 'quality_of_faculty', 'publications', 'influence',

       'citations', 'broad_impact', 'patents', 'score']

scaler = MinMaxScaler()

df_train[vars] = scaler.fit_transform(df_train[vars])
df_train.shape

df_train.head()
y_train = df_train.pop('world_rank')

X_train = df_train
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):

    vif = pd.DataFrame()

    X = df

    vif['features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'],2)    

    vif = vif.sort_values(by = 'VIF',ascending=False)

    return vif
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
from sklearn.feature_selection import RFE

lm.fit(X_train,y_train)

rfe = RFE(lm,15)

rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
cols = X_train.columns[rfe.support_]

cols
X_train_rfe = X_train[cols]
X_train_rfe.head()
# Model 1

import statsmodels.api as sm

X_train_lm1 = sm.add_constant(X_train_rfe)

lm1 = sm.OLS(y_train,X_train_lm1).fit()

print(lm1.summary())
# Model 2

import statsmodels.api as sm

X_train_new2 = X_train_rfe.drop('Singapore',axis='columns')

X_train_lm2 = sm.add_constant(X_train_new2)

lm2 = sm.OLS(y_train,X_train_lm2).fit()

print(lm2.summary())
# Model 3

X_train_new3 = X_train_new2.drop('Colombia',axis='columns')

X_train_lm3 = sm.add_constant(X_train_new3)

lm3 = sm.OLS(y_train,X_train_lm3).fit()

print(lm3.summary())
vif1 = X_train_lm3.drop('const',axis='columns')

calculate_vif(vif1)
# Model 4

import statsmodels.api as sm

X_train_new4 = X_train_new3.drop('publications',axis='columns')

X_train_lm4 = sm.add_constant(X_train_new4)

lm4 = sm.OLS(y_train,X_train_lm4).fit()

print(lm4.summary())
vif2 = X_train_lm4.drop('const',axis=1)

calculate_vif(vif2)
# Model 5

import statsmodels.api as sm

X_train_new5 = X_train_new4.drop('quality_of_education',axis='columns')

X_train_lm5 = sm.add_constant(X_train_new5)

lm5 = sm.OLS(y_train,X_train_lm5).fit()

print(lm5.summary())
vif3 = X_train_lm5.drop('const',axis=1)

calculate_vif(vif3)
# Model 6

import statsmodels.api as sm

X_train_new6 = X_train_new5.drop('patents',axis='columns')

X_train_lm6 = sm.add_constant(X_train_new6)

lm6 = sm.OLS(y_train,X_train_lm6).fit()

print(lm6.summary())
# Model 7

import statsmodels.api as sm

X_train_new7 = X_train_new6.drop('score',axis='columns')

X_train_lm7 = sm.add_constant(X_train_new7)

lm7 = sm.OLS(y_train,X_train_lm7).fit()

print(lm7.summary())
vif4 = X_train_lm7.drop('const',axis=1)

calculate_vif(vif4)
y_train_rank = lm7.predict(X_train_lm7)
from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
fig = plt.figure()

sns.distplot(y_train_rank-y_train,bins=20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  

fig.show()
vars1 = ['world_rank', 'national_rank', 'quality_of_education',

       'alumni_employment', 'quality_of_faculty', 'publications', 'influence',

       'citations', 'broad_impact', 'patents', 'score']

df_test[vars1] = scaler.transform(df_test[vars1])
y_test = df_test.pop('world_rank')

X_test = df_test
X_test_new = X_test[X_train_new7.columns]
X_test_new = sm.add_constant(X_test_new)
y_pred = lm7.predict(X_test_new)
fig = plt.figure()

plt.scatter(y_pred,y_test)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)    
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)