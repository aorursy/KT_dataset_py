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
pd.options.display.float_format = '{:.1f}'.format

pd.set_option('display.max_rows', 1000)

pd.get_option("display.max_columns", 1000)

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.describe().transpose()
# before tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
df.head(10)
df["bathrooms"] = df['bathrooms'].round(0).astype(int)

df["floors"] = df['floors'].round(0).astype(int)



df['Is_Basement'] = [1 if i != 0 else 0 for i in df['sqft_basement']]

df['Is_Bathrooms'] = [1 if i != 0 else 0 for i in df['bathrooms']]

df['Is_Bedrooms'] = [1 if i != 0 else 0 for i in df['bedrooms']]

df['Was_Renovated'] = [1 if i != 0 else 0 for i in df['yr_renovated']]

df['More_than_1_floor'] = [1 if i > 1 else 0 for i in df['floors']]

df['grade'] = [1 if i >= 11 else 3 if i<=3 else 2 for i in df['grade']]





for i in range(0, len(df)):

    if df['yr_renovated'][i]!=0:

        df['yr_built'][i]=df['yr_renovated'][i]

        

# just take the year from the date column

df['sales_yr']=df['date'].astype(str).str[:4]

df['sales_yr']=df['sales_yr'].astype(int)



df['age'] = df['sales_yr'] - df['yr_built'] 

df['sqft_living_per_lot'] = df['sqft_living']/df['sqft_lot']

df['sqft_living_per_floor'] = df['sqft_living']/df['floors']



df['sqft_living'] = np.log(df.sqft_living + 0.01)

df['sqft_lot'] = np.log(df.sqft_lot + 0.01)

df['sqft_living15'] = np.log(df.sqft_living15 + 0.01)

df['sqft_living_per_floor'] = np.log(df.sqft_living_per_floor + 0.01)

# df['price'] = np.log(df.price + 0.01)





        

df = df.drop(['id', 'long', 'lat', 'yr_renovated', 'date','yr_built', 'sqft_basement', 'sqft_above', 'sales_yr', 'sqft_lot15'], axis=1)



categorial_cols = ['floors', 'view', 'condition', 'grade']



for cc in categorial_cols:

    dummies = pd.get_dummies(df[cc], drop_first=False)

    dummies = dummies.add_prefix("{}#".format(cc))

    df.drop(cc, axis=1, inplace=True)

    df = df.join(dummies)

    

dummies_zipcodes = pd.get_dummies(df['zipcode'], drop_first=False)

dummies_zipcodes.reset_index(inplace=True)

dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))

# dummies_zipcodes = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]

df.drop('zipcode', axis=1, inplace=True)

df = df.join(dummies_zipcodes)

df.head()
import seaborn as sns 

sns.distplot(np.log1p(df['price']))
# after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
df.age.max()
def spearman(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['price'], 'spearman') for f in features]

    spr = spr.sort_values('spearman')

    plt.figure(figsize=(6, 0.25*len(features)))

    f, ax = plt.subplots(figsize=(12, 9))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    

features = df.drop(['price'], axis = 1).columns

spearman(df, features)
# corrMatrix = df.corr()

# f, ax = plt.subplots(figsize=(20, 10))

# sns.heatmap(corrMatrix, annot = True)

# plt.show()
from collections import Counter

num_col = df.columns

# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR 

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(df,2, num_col)

df.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers

df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

print('Outliers dropped')
X = df.drop(['price'], axis=1)

y = df.price
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=29)





d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)









params = {

        'objective':'reg:linear',

        'n_estimators': 10,

        'booster':'gbtree',

        'max_depth':2,

        'eval_metric':'rmse',

        'learning_rate':0.1, 

        'min_child_weight':3,

        'subsample':0.85,

        'colsample_bytree':0.5,

        'seed':45,

        'reg_alpha':1,#1e-03,

        'reg_lambda':1,

        'gamma':0,

        'nthread':-1



}





watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=400, maximize=False, verbose_eval=10)

d_test = clf.predict(d_valid)

r2_score(y_valid, d_test)
dmatrix_data = xgb.DMatrix(data=X, label=y)



cv_params = {

        'objective':'reg:linear',

        'n_estimators': 10,

        'booster':'gbtree',

        'max_depth':2,

        'eval_metric':'rmse',

        'learning_rate':0.1, 

        'min_child_weight':3,

        'subsample':0.85,

        'colsample_bytree':0.5,

        'seed':45,

        'reg_alpha':1,#1e-03,

        'reg_lambda':1,

        'gamma':0,

        'nthread':-1

}

cross_val = xgb.cv(

    params=cv_params,

    dtrain=dmatrix_data, 

    nfold=5,

    num_boost_round=5000, 

    early_stopping_rounds=1000, 

    metrics='rmse', 

    as_pandas=True, 

    seed=29)

print(cross_val.tail(1))