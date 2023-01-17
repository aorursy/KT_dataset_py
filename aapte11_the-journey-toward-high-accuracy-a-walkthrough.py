import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.style as style

style.use('seaborn-whitegrid')

import seaborn as sns





pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)

train_feature_url = '../input/train_features.csv'

test_feature_url = '../input/test_features.csv'

train_labels_url = '../input/train_labels.csv'



train_F = pd.read_csv(train_feature_url, index_col='id')

test_F = pd.read_csv(test_feature_url, index_col='id')

train_L = pd.read_csv(train_labels_url, index_col='id')



train_F.shape, test_F.shape, train_L.shape  #,  train_F_C.shape, test_F_C.shape, # 
train_F.dtypes
null_list = []



for col in train_F.columns:

  if train_F[col].isnull().sum() > 0:

    null_list.append(col)

    

null_list.append('recorded_by')

null_list.append('date_recorded')

    

null_list
def convert_datetime(df, col):

  df[col] = pd.to_datetime(df[col])

  df['day_of_week'] = df[col].dt.weekday_name 

  df['year'] = df[col].dt.year

  df['month'] = df[col].dt.month 

  df['day'] = df[col].dt.day 

  

  return None



convert_datetime(train_F, 'date_recorded')

convert_datetime(test_F, 'date_recorded')



train_F.dtypes
train_F['region_code'] = train_F['region_code'].astype('category')

test_F['region_code'] = test_F['region_code'].astype('category')

train_F['district_code'] = train_F['district_code'].astype('category')

test_F['district_code'] = test_F['district_code'].astype('category')

train_F['wpt_name'] = train_F['wpt_name'].astype('category')

test_F['wpt_name'] = test_F['wpt_name'].astype('category')

train_F['ward'] = train_F['ward'].astype('category')

test_F['ward'] = test_F['ward'].astype('category')





train_F.dtypes
def df_split(df):

  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

  df_num = df.select_dtypes(include=numerics)

  df_cat = df.drop(df_num, axis = 'columns')

  print (df.shape, df_num.shape, df_cat.shape)

  return df_num, df_cat

  
train_F_num, train_F_cat = df_split(train_F)



test_F_num, test_F_cat = df_split(test_F)
train_F_num.describe().T
train_F_num['construction_year'].loc[train_F_num['construction_year'] == 0] = train_F_num['year']

test_F_num['construction_year'].loc[test_F_num['construction_year'] == 0] = test_F_num['year']



train_F_num.describe().T
mean_lat_train = train_F_num['latitude'].mean()

mean_long_train = train_F_num['longitude'].mean()

mean_lat_test = test_F_num['latitude'].mean()

mean_long_test = test_F_num['longitude'].mean()





train_F_num['distance'] = np.sqrt((train_F_num['longitude'] - mean_long_train)**2 + (train_F_num['latitude'] - mean_lat_train)**2)

test_F_num['distance'] = np.sqrt((test_F_num['longitude'] - mean_long_test)**2 + (test_F_num['latitude'] - mean_lat_test)**2)



train_F_num['distance3d'] = np.sqrt((train_F_num['gps_height']**2 + train_F_num['longitude'] - mean_long_train)**2 + (train_F_num['latitude'] - mean_lat_train)**2)

test_F_num['distance3d'] = np.sqrt((test_F_num['gps_height']**2 + test_F_num['longitude'] - mean_long_test)**2 + (test_F_num['latitude'] - mean_lat_test)**2)





train_F_num.describe().T
for col in train_F_cat.columns:

  print (col, train_F_cat[col].nunique())
null_list
cols_kept = []



for col in train_F_cat.columns:

  if col not in null_list:

    if train_F_cat[col].nunique() <= 125:

      cols_kept.append(col)

    

print (len(cols_kept))

    

cols_kept



small_cat_train = train_F_cat[cols_kept]

small_cat_test = test_F_cat[cols_kept]



small_cat_train.shape, small_cat_test.shape
def dummy_df(category_df):

  df_dummy = pd.DataFrame()

  for col in category_df.columns:

    df_dummy = pd.concat([df_dummy, pd.get_dummies(category_df[col], drop_first=True, prefix = 'Is')], axis='columns')

  return df_dummy
df_dumb_train = dummy_df(small_cat_train)

df_dumb_test = dummy_df(small_cat_test)
df_dumb_train.shape, df_dumb_test.shape
a = list(df_dumb_train.columns.values)



print(a)

print(len(a))
b = list(df_dumb_test.columns.values)



print(b)

print(len(b))
a == b
def ex_cols(a,b):

  ex_a = []

  ex_b = []

  for i in range(0,len(a)):

    if a[i] not in b:

      ex_a.append(a[i])

  for j in range(0,len(b)):

    if b[j] not in a:

      ex_b.append(b[j])

  return ex_a, ex_b



ex_a, ex_b = ex_cols(a,b)



ex_a,ex_b
for col in df_dumb_train.columns:

  if col in ex_a:

    del df_dumb_train[col]



for col in df_dumb_test.columns:

  if col in ex_b:

    del df_dumb_test[col]



df_dumb_train.shape, df_dumb_test.shape
c = list(df_dumb_train.columns.values)

d = list(df_dumb_test.columns.values)



c == d
for i in range(0,len(c)):

  if c[i] != d[i]:

    print("No match")

    
X_train = pd.concat([train_F_num,df_dumb_train],axis='columns')

X_test = pd.concat([test_F_num,df_dumb_test],axis='columns')



X_train.shape, X_test.shape
X_train.head()
np.any(np.isnan(X_train)), np.any(np.isnan(X_test))

train_L.head()
train_L['status_group'] = train_L['status_group'].astype('category')



train_L.dtypes, train_L.shape
y_train = train_L['status_group']



y_train.value_counts()
majority_class = y_train.mode()[0]



print(majority_class)



y_pred = pd.DataFrame(np.full(shape=len(X_test), fill_value = majority_class))

temp = X_test.reset_index()



y_pred['id'] = temp['id'].values

y_pred.rename(columns={0:'status_group'}, inplace=True)

y_pred.set_index('id', inplace=True)



y_pred.head()



print(y_pred.shape)
# from google.colab import files



# y_pred.to_csv('majority_class.csv')

# files.download('majority_class.csv')
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier





clf_rf = RandomForestClassifier(n_estimators=100, 

                                max_depth=34,

                                min_samples_split = 17,

                                min_samples_leaf = 1,

                                criterion = 'gini', 

                                max_features = 6, 

                                oob_score = True, 

                                random_state=237)



clf_lr = LogisticRegression(random_state=237, solver='lbfgs', multi_class='multinomial', max_iter=1000)



def quick_eval(X,y, clf):

  from sklearn.model_selection import train_test_split

  from sklearn.preprocessing import StandardScaler

  from sklearn.preprocessing import RobustScaler

  from sklearn.metrics import accuracy_score



  

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state=237)

  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

  

  scaler = StandardScaler()

#   scaler = RobustScaler()

  X_train_clf = scaler.fit_transform(X_train)

  X_test_clf = scaler.transform(X_test)

  

  clf.fit(X_train_clf, y_train)

  

  y_pred_train = clf.predict(X_train_clf)

  

  y_pred = clf.predict(X_test_clf)

  

  

  return accuracy_score(y_train,y_pred_train), accuracy_score(y_test, y_pred)
# %%time



quick_eval(X_train, y_train, clf_rf)
# %%time



# from sklearn.model_selection import RandomizedSearchCV

# from scipy.stats import randint as sp_randint

# # parameters for GridSearchCV

# # specify parameters and distributions to sample from

# param_dist = {"n_estimators": [100, 200,300],

#               "max_features": sp_randint(5, 9),

#               "max_depth": [18,22,26,30,34,38],

#               "min_samples_split": sp_randint(8, 32),

#               "min_samples_leaf": sp_randint(1, 20)              

#              }

# # run randomized search

# n_iter_search = 200

# random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist,

#                                    n_iter=n_iter_search)



# random_search.fit(X_train, y_train)
# def report(results, n_top=5):

#     for i in range(1, n_top + 1):

#         candidates = np.flatnonzero(results['rank_test_score'] == i)

#         for candidate in candidates:

#             print("Model with rank: {0}".format(i))

#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

#                   results['mean_test_score'][candidate],

#                   results['std_test_score'][candidate]))

#             print("Parameters: {0}".format(results['params'][candidate]))

#             print("")

            

# report(random_search.cv_results_)            
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MaxAbsScaler



scaler = StandardScaler()

# scaler = MinMaxScaler()

# scaler = RobustScaler()

# scaler = MaxAbsScaler()



X_train_clf = scaler.fit_transform(X_train)

X_test_clf = scaler.transform(X_test)





X_train_clf.shape, y_train.shape, X_test_clf.shape
%%time



from sklearn.model_selection import cross_validate



from sklearn.metrics import accuracy_score





scores = cross_validate(clf_rf,

                        X_train_clf,y_train, 

                        scoring = 'accuracy', cv=5) 



pd.DataFrame(scores)
def predictor(X_train, X_test, y_train, clf):

  from sklearn.preprocessing import StandardScaler

  from sklearn.preprocessing import RobustScaler

 

  from sklearn.metrics import accuracy_score

  



  y_pred = pd.DataFrame()

  

  temp_test = X_test.reset_index()

  y_id = temp_test['id']

#   scaler = StandardScaler()

  scaler = RobustScaler()

  

  X_train_clf = scaler.fit_transform(X_train)



  X_test_clf = scaler.transform(X_test)

  clf.fit(X_train_clf, y_train)

  

  y_pred_train = clf.predict(X_train_clf)

  

  print (f'\nThe accuracy score of the training set is {round(accuracy_score(y_train, y_pred_train), 5)}\n')

  

  prediction = pd.DataFrame(clf.predict(X_test_clf))

  

  y_pred = pd.concat([y_id, prediction], axis='columns')



  y_pred.rename(columns={0:'status_group'}, inplace=True)

  

  y_pred.set_index('id', inplace=True)

  

  return y_pred
%%time



df = predictor(X_train, X_test, y_train, clf_rf)



df.head()
df['status_group'].value_counts()
df.shape
# from google.colab import files



# df.to_csv('submission.csv')

# files.download('submission.csv')