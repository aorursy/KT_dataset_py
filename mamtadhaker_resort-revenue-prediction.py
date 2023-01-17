# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, FeaturesData, Pool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# ,parse_dates=['booking_date', 'checkin_date', 'checkout_date']

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
data = train.append(test,ignore_index = True) 

print(data.shape)

data.head()
data.info()
data.describe()
data.isna().sum()
data.nunique()
print(data['booking_date'].max())

print(data['booking_date'].min())
data.head()
def feature_engineering(df):

    

    df.loc[:,'booking_date'] = pd.to_datetime(df['booking_date'], format="%d/%m/%y",infer_datetime_format=True)

    df.loc[:,'checkin_date'] = pd.to_datetime(df['checkin_date'], format="%d/%m/%y",infer_datetime_format=True)

    df.loc[:,'checkout_date'] = pd.to_datetime(df['checkout_date'], format="%d/%m/%y",infer_datetime_format=True)

    df.loc[:,'trip_length'] = df['checkout_date']- df['checkin_date']

    df.loc[:,'days_before_planning'] = df['checkin_date'] - df['booking_date']

    df.loc[:,'trip_length'] = df['trip_length'].apply(lambda x : x.days)

    df.loc[:,'days_before_planning'] = df['days_before_planning'].apply(lambda x : x.days)

    trip_count = data.groupby(['memberid'])['reservation_id'].agg(['count'])

    df.loc[:,'trip_count'] = data['memberid'].apply(lambda i: trip_count.loc[i][0])

    return df

    

# def drop(df):

#     to_drop = ['reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']

#     return df.drop(to_drop,axis=1)



def create_data(df):

    df = feature_engineering(df)

#     df = drop(df)

    return df
# trip_count = data.groupby(['memberid'])['reservation_id'].agg(['count'])

# trip_count = pd.DataFrame({'memberid':trip_count.index, 'trip_count':trip_count['count'].values})
dataset = create_data(data)
dataset['days_before_planning'] = dataset['days_before_planning'].apply(lambda x : x if x>=0 else 0)
dataset.columns
dataset.head()
# dataset[dataset['days_before_planning']==0][['booking_date', 'checkin_date', 'checkout_date','roomnights','trip_length', 'days_before_planning']]
# Function to determine if column in dataframe is string.

def is_str(col):

    for i in col:

        if pd.isnull(i):

            continue

        elif isinstance(i, str):

            return True

        else:

            return False

# Splits the mixed dataframe into categorical and numerical features.

def split_features(df):

    cfc = []

    nfc = []

    for column in df:

        if is_str(df[column]):

            cfc.append(column)

        else:

            nfc.append(column)

    return df[cfc], df[nfc]
def preprocess(cat_features, num_features):

    cat_features = cat_features.fillna("None")

    for column in num_features:

        num_features[column].fillna(np.nanmean(num_features[column]), inplace=True)

    return cat_features, num_features
y_train = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]['amount_spent_per_room_night_scaled']

to_drop=['amount_spent_per_room_night_scaled','reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']

X_train = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True].drop(to_drop, axis=1)

X_test = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True].drop(['reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date'],axis=1)

# dftrain=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]

# dftest=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True]

# dftrain.head()
# Apply the "split_features" function on the data.

cat_tmp_train, num_tmp_train = split_features(X_train)

cat_tmp_test, num_tmp_test = split_features(X_test)
# Now to apply the "preprocess" function.

# Getting a "SettingWithCopyWarning" but I usually ignore it.

cat_features_train, num_features_train = preprocess(cat_tmp_train, num_tmp_train)

cat_features_test, num_features_test = preprocess(cat_tmp_test, num_tmp_test)
train_pool = Pool(

    data = FeaturesData(num_feature_data = np.array(num_features_train.values, dtype=np.float32), 

                    cat_feature_data = np.array(cat_features_train.values, dtype=object), 

                    num_feature_names = list(num_features_train.columns.values), 

                    cat_feature_names = list(cat_features_train.columns.values)),

    label =  np.array(y_train, dtype=np.float32)

)
test_pool = Pool(

    data = FeaturesData(num_feature_data = np.array(num_features_test.values, dtype=np.float32), 

                    cat_feature_data = np.array(cat_features_test.values, dtype=object), 

                    num_feature_names = list(num_features_test.columns.values), 

                    cat_feature_names = list(cat_features_test.columns.values))

)
model = CatBoostRegressor(iterations=3000,loss_function = 'RMSE', learning_rate=0.05, depth=5)

# Fit model

model.fit(train_pool)

# Get predictions

preds = model.predict(test_pool)
res_id = test['reservation_id']
df = pd.DataFrame({'reservation_id': res_id, 'amount_spent_per_room_night_scaled': preds}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

df.to_csv("submission.csv", index=False)
# X,y=dftrain.drop('loan_default',axis=1),dftrain['loan_default']

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state = 1994)


# categorical_features_indices = np.where(X_train.dtypes =='object')[0]

# categorical_features_indices
# import catboost



# class ModelOptimizer:

#     best_score = None

#     opt = None

    

#     def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

#         self.model = model

#         self.X_train = X_train

#         self.y_train = y_train

#         self.categorical_columns_indices = categorical_columns_indices

#         self.n_fold = n_fold

#         self.seed = seed

#         self.early_stopping_rounds = early_stopping_rounds

#         self.is_stratified = is_stratified

#         self.is_shuffle = is_shuffle

        

        

#     def update_model(self, **kwargs):

#         for k, v in kwargs.items():

#             setattr(self.model, k, v)

            

#     def evaluate_model(self):

#         pass

    

#     def optimize(self, param_space, max_evals=10, n_random_starts=2):

#         start_time = time.time()

        

#         @use_named_args(param_space)

#         def _minimize(**params):

#             self.model.set_params(**params)

#             return self.evaluate_model()

        

#         opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

#         best_values = opt.x

#         optimal_values = dict(zip([param.name for param in param_space], best_values))

#         best_score = opt.fun

#         self.best_score = best_score

#         self.opt = opt

        

#         print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

#         print('updating model with optimal values')

#         self.update_model(**optimal_values)

#         plot_convergence(opt)

#         return optimal_values

    

# class CatboostOptimizer(ModelOptimizer):

#     def evaluate_model(self):

#         validation_scores = catboost.cv(

#         catboost.Pool(self.X_train, 

#                       self.y_train, 

#                       cat_features=self.categorical_columns_indices),

#         self.model.get_params(), 

#         nfold=self.n_fold,

#         stratified=self.is_stratified,

#         seed=self.seed,

#         early_stopping_rounds=self.early_stopping_rounds,

#         shuffle=self.is_shuffle,

#         verbose=100,

#         plot=False)

#         self.scores = validation_scores

#         test_scores = validation_scores.iloc[:, 2]

#         best_metric = test_scores.max()

#         return 1 - best_metric
# from skopt import gp_minimize

# from skopt.space import Real, Integer

# from skopt.utils import use_named_args

# from skopt.plots import plot_convergence

# import time
# cb = catboost.CatBoostClassifier(n_estimators=4000, # use large n_estimators deliberately to make use of the early stopping

#                          loss_function='Logloss',

#                          eval_metric='AUC',

#                          boosting_type='Ordered', # use permutations

#                          random_seed=1994, 

#                          use_best_model=True)

# cb_optimizer = CatboostOptimizer(cb, X_train, y_train,categorical_columns_indices=categorical_features_indices)

# params_space = [Real(0.01, 0.8, name='learning_rate'),]

# cb_optimal_values = cb_optimizer.optimize(params_space)