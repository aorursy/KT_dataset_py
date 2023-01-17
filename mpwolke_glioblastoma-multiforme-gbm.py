# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns #visualization

import matplotlib.pyplot as plt #visualization

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/glioblastomas/GBMs(3).csv', encoding='ISO-8859-2')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
# Let's find the null values in data



total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
# Save target value for later

y = df.glioma_dx.values



# In order to make imputing easier, we combine train and test data

df.drop(['glioma_dx'], axis=1, inplace=True)

df = pd.concat((df, test)).reset_index(drop=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.model_selection import KFold

# Indicate number of folds for cross validation

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



# Parameters for models

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LassoCV

# Lasso Model

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas2, random_state = 42, cv=kfolds))



# Printing Lasso Score with Cross-Validation

lasso_score = cross_val_score(lasso, X, y, cv=kfolds, scoring='neg_mean_squared_error')

lasso_rmse = np.sqrt(-lasso_score.mean())

print("LASSO RMSE: ", lasso_rmse)

print("LASSO STD: ", lasso_score.std())
# Training Model for later

lasso.fit(X_train, y_train)
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



surgery_year = df.surgery_year.values

glioma_egfr = df.glioma_egfr.values

glioma_mgmt = df.glioma_mgmt.values



sns.distplot(surgery_year , ax = ax[0] , color = 'blue').set_title('GBM Surgery Year' , fontsize = 14)

sns.distplot(glioma_egfr , ax = ax[1] , color = 'cyan').set_title('GBM Glioma EGFR' , fontsize = 14)

sns.distplot(glioma_mgmt , ax = ax[2] , color = 'purple').set_title('GBM Glioma MGMT ' , fontsize = 14)





plt.show()
import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,6))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(df['glioma_mgmt'].skew())) 

print("Kurtosis: " + str(df['glioma_mgmt'].kurt()))

plotting_3_chart(df, 'glioma_mgmt')
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
#Codes from Mario Filho https://www.kaggle.com/mariofilho/live26-https-youtu-be-zseefujo0zq



from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['brain_location___13']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['surgery_year']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df['progression_year'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('Progression Year Distribution & Last alive Year', size=18)

ax.set_ylabel('Progression Year Dstribution', size=10)

ax.set_xlabel('count', size=10)
fig=sns.lmplot(x='surgery_year', y="last_alive_year",data=df) 
ax = df.groupby('surgery_year')['glioma_dx_other', 'progression_year'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Progression after Surgery Year')

plt.xlabel('Surgery Year')

plt.ylabel('Rate of Glioma Other than DX & Progression Year')



plt.show()
ax = df.groupby('surgery_year')['brain_location___19', 'last_alive_year'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Last Alive Year & Brain Location 19 After Surgery', logx=True, linewidth=3)

plt.xlabel('Count Log')

plt.ylabel('Rate of Alive & Brain Location 19')

plt.show()