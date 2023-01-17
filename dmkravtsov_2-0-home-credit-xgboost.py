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
pd.set_option('display.max_rows', 1000)
train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

train.head()
## not improve our score

# from collections import Counter

# num_col = train.loc[:,'NAME_CONTRACT_TYPE':'AMT_REQ_CREDIT_BUREAU_YEAR'].select_dtypes(exclude=['object']).columns

# # Outlier detection 



# def detect_outliers(df,n,features):

#     """

#     Takes a dataframe df of features and returns a list of the indices

#     corresponding to the observations containing more than n outliers according

#     to the Tukey method.

#     """

#     outlier_indices = []

    

#     # iterate over features(columns)

#     for col in features:

#         # 1st quartile (25%)

#         Q1 = np.percentile(df[col], 25)

#         # 3rd quartile (75%)

#         Q3 = np.percentile(df[col],75)

#         # Interquartile range (IQR)

#         IQR = Q3 - Q1

        

#         # outlier step

#         outlier_step = 1.5 * IQR

        

#         # Determine a list of indices of outliers for feature col

#         outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

#         # append the found outlier indices for col to the list of outlier indices 

#         outlier_indices.extend(outlier_list_col)

        

#     # select observations containing more than 2 outliers

#     outlier_indices = Counter(outlier_indices)        

#     multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

#     return multiple_outliers   



# # detect outliers from Age, SibSp , Parch and Fare

# Outliers_to_drop = detect_outliers(train,2, num_col)

# train.loc[Outliers_to_drop] # Show the outliers rows

# # Drop outliers

# train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df = pd.concat((train.loc[:,'NAME_CONTRACT_TYPE':'AMT_REQ_CREDIT_BUREAU_YEAR'], test.loc[:,'NAME_CONTRACT_TYPE':'AMT_REQ_CREDIT_BUREAU_YEAR']))
# before tuning



def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)


# simplest NaN imputation



for col in df:

    if df[col].dtype == 'object':

        df[col].fillna('N')

    df[col].fillna(10000, inplace=True) ## one of the best result (0.7599 vs 0.75760 of final result)

    

# # polynomial features adding (is not good idea finally)

# from sklearn.preprocessing import PolynomialFeatures

# poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_ANNUITY']] ## first 5 important features as per baseline model (therefore adding new 56 polynomial  features), but finally it's not give positive result (0.7583 vs 0.75911)

# poly_transformer = PolynomialFeatures(degree = 3)

# poly_transformer.fit(poly_features)

# poly_features = poly_transformer.transform(poly_features)

# # # # Scale the polynomial features

# # from sklearn.preprocessing import MinMaxScaler

# # scaler = MinMaxScaler(feature_range = (0, 1))

# # poly_features = scaler.fit_transform(poly_features)

# # Put poly features into dataframe

# poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_ANNUITY']))

# df = df.merge(poly_features)

  

# making category features from some numerical (part 1 - for short shape numerical features)



for col in df:

    if df[col].nunique()<=30:

        df[col] = df[col].astype(str)

        

# # making category features from some numerical (part 2 - for long shape numerical features) - not help us to increase score

# num_col = df.select_dtypes(exclude=['object']).columns

# for col in num_col:

#     if df[col].nunique()>30:

#         df[col+str('_category')] = pd.qcut(df[col].rank(method='first'),q=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],labels=False,precision=1).astype(str)

        



# simplest encoding for object columns in df



df = pd.get_dummies(df) ## therefore adding  additional feature
# after tuning



def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100,2)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
df.shape
# # it's gives us negative (abt -0.004) result to baseline

# def descriptive_stat_feat(df):

#     df = pd.DataFrame(df)

#     dcol= [c for c in df.columns if df[c].nunique()>=3]

#     d_median = df[dcol].median(axis=0)

#     d_mean = df[dcol].mean(axis=0)

#     q1 = df[dcol].apply(np.float32).quantile(0.25)

#     q3 = df[dcol].apply(np.float32).quantile(0.75)

    

#     #Add mean and median column to data set having more then 3 categories

#     for c in dcol:

#         df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

#         df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

#         df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

#         df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

#     return df



# df = descriptive_stat_feat(df)
#creating matrices for feature selection:

X_train = df[:train.shape[0]]

X_test_fin = df[train.shape[0]:]

y = train.TARGET

X_train['Y'] = y

df = X_train



X = df.drop('Y', axis=1)

y = df.Y
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test_fin)



params = {

        'objective':'binary:logistic',

        'tree_method':'gpu_hist',

        'eta': 0.3,

        'max_depth':6,

        'learning_rate':0.01,

        'eval_metric':'auc',

        'min_child_weight':2,

        'subsample':0.8,

        'colsample_bytree':0.7,

        'seed':29,

        'reg_lambda':0.8,

        'reg_alpha':0.000001,

        'gamma':0.1,

        'scale_pos_weight':1,

        'n_estimators': 500,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

nrounds=10000 

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=50, 

                           maximize=True, verbose_eval=10)

p_test = model.predict(d_test)
sub = pd.DataFrame({'SK_ID_CURR':test['SK_ID_CURR'],'TARGET':p_test})



sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig,ax = plt.subplots(figsize=(100,75))

xgb.plot_importance(model,ax=ax,height=0.8,color='r')

#plt.tight_layout()

plt.show()