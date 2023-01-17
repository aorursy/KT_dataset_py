from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt
def seed_everything(seed=0):

    np.random.seed(seed)



seed_everything()
# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#TRAIN

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

#train_identity = train_identity.dropna(thresh=10)

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

#train_transaction = train_transaction.dropna(thresh=10)
# TEST

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

#test_identity = train_identity.dropna(thresh=10)

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

#test_transaction = test_transaction.dropna(thresh=10)
# TRAIN

# Reduce the number of zeros since it is imbalanced

train_transaction_0 = train_transaction[train_transaction['isFraud'] == 0].sample(frac=0.05)

train_transaction_1 = train_transaction[train_transaction['isFraud'] == 1]

train_trans_reduced = pd.concat([train_transaction_0, train_transaction_1])







train = train_trans_reduced.merge(train_identity,how='left', left_index=True, right_index=True)

train_y = train['isFraud']



# TEST

test_x = test_transaction.merge(test_identity,how='left', left_index=True, right_index=True)
train = train.reset_index()

test_x = test_x.reset_index()
rm_cols = ['TransactionID','TransactionDT','isFraud']



# Drop unnecessary columns

for col in rm_cols:

    train = train.drop(col, axis=1)

    if col != "isFraud":

        test_x = test_x.drop(col, axis=1)
train_transaction['isFraud'].value_counts().plot(kind='bar', title="Before cleaning")
train_trans_reduced['isFraud'].value_counts().plot(kind='bar', title="After cleaning")
del train_transaction

del train_transaction_0

del train_transaction_1

del train_identity



del test_transaction

del test_identity
p = 'P_emaildomain'

r = 'R_emaildomain'

uknown = 'email_not_provided'



for df in [train, test_x]:

    df[p] = df[p].fillna(uknown)

    df[r] = df[r].fillna(uknown)

    

    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=uknown),1,0)



    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])

    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])
for col in list(train):

    if train[col].dtype=='O' or train[col].dtype=='object':

        print(col)

        train[col] = train[col].fillna('unseen_before_label')

        test_x[col]  = test_x[col].fillna('unseen_before_label')

        

        le = preprocessing.LabelEncoder()

        le.fit(list(train[col])+list(test_x[col]))

        train[col] = le.transform(train[col])

        test_x[col]  = le.transform(test_x[col])

        

        train[col] = train[col]

        test_x[col] = test_x[col]
# Reduce memory usage

train_x = reduce_mem_usage(train)

test_x = reduce_mem_usage(test_x)
# Creando conjuntos de validacion y test

seed = 7

test_size = 0.35

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=seed)
params ={

        'booster':'gbtree', 

        'objective':'binary:logistic', 

        'n_estimators':10000,

        'rate_drop': 0.2 

        }
my_model = XGBClassifier(**params)
my_model.fit(

            X_train, 

            y_train, 

            eval_set=[(X_train, y_train), (X_val, y_val)], 

            verbose=30, 

            eval_metric='auc',

            early_stopping_rounds=40

            )
predictions = my_model.predict_proba(test_x)[:, 1]
def show_important_features(names, importances, limit, return_columns=False):

    impo_list = zip(names, importances)

    impo_list = sorted(impo_list, key=lambda t: t[1], reverse=True)

    

    sorted_importances = []

    sorted_names = []

    for idx, el in enumerate(impo_list):

        if idx < limit:

            sorted_names.append(el[0])

            sorted_importances.append(el[1])

    

    if not return_columns:

        fig, ax = plt.subplots(figsize=(9,7))

        

        ax.barh(sorted_names, sorted_importances, align='center')

        ax.invert_yaxis()

        ax.set_xlabel('Feature importance')

        ax.set_title('Most influencial {} features'.format(limit))

        

        plt.show()

    else:

        return sorted_names
names = X_train.columns

importances = my_model.feature_importances_

limit = 20



show_important_features(names, importances, limit)
# Creando conjuntos de validacion y test

most_important_80 = show_important_features(names, importances, 80, return_columns=True)
new_train_x = train_x[train_x.columns & most_important_80]
seed = 7

test_size = 0.20

X_train, X_val, y_train, y_val = train_test_split(new_train_x, train_y, test_size=test_size, random_state=seed)
params ={

        'booster':'gbtree', 

        'objective':'binary:logistic', 

        'n_estimators':10000

        }
my_model_importance = XGBClassifier(**params)
my_model_importance.fit(

            X_train, 

            y_train, 

            eval_set=[(X_train, y_train), (X_val, y_val)], 

            verbose=30, 

            eval_metric='auc',

            early_stopping_rounds=40

            )
new_test_x = test_x[test_x.columns & most_important_80]
predictions_importance = my_model_importance.predict_proba(new_test_x)[:, 1]
names = X_train.columns

importances = my_model_importance.feature_importances_

limit = 20



show_important_features(names, importances, limit)
final_predictions = (predictions_importance + predictions) / 2
submit = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

submit['isFraud'] = final_predictions

submit.head()
submit.to_csv('submission.csv', index=False)