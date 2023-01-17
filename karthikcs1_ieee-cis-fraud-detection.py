import numpy as np 

import pandas as pd 

from sklearn.feature_selection import SelectKBest, chi2,f_classif

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

from sklearn.model_selection import train_test_split
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





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
! ls ../input/ieee-fraud-detection/
#!ls ../input/ieee-fraud-detection

df_transacation = import_data('../input/ieee-fraud-detection/train_transaction.csv')

df_identity = import_data('../input/ieee-fraud-detection/train_identity.csv')

print(df_transacation.shape, df_identity.shape)
df_transacation.head()

df_identity.head()
df_identity.set_index(['TransactionID'], inplace=True)

def get_id(x):

#     return df_identity[df_identity['TransactionID'] == x].shape[0]

    try :

        df_identity.loc[x].shape

        return 1

    except :

        return 0

    

# get_id(2987004)
df_transacation['id_exists'] = df_transacation['TransactionID'].apply(get_id)
'Identity records exists in {:.2f}% of transactions'.format(df_transacation[df_transacation['id_exists'] == 1].shape[0] / df_transacation.shape[0]*100)
print('Fradulent transactions are {:.2f}%'.format(df_transacation[df_transacation['isFraud'] == 1].shape[0] / df_transacation.shape[0]*100))

print('with {} Fradulent transactions'.format(df_transacation[df_transacation['isFraud'] == 1].shape[0]))


# gives some infos on columns types and number of null values

tab_info=pd.DataFrame(df_transacation.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df_transacation.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df_transacation.isnull().sum()/df_transacation.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

tab_info = tab_info.transpose()

tab_info
'There are {} columns which have  more than {}% null values'.format(tab_info[tab_info['null values (%)'] > 30].shape[0], 30)
cols_to_remove = tab_info[tab_info['null values (%)'] > 30].index.values

cols_to_remove

df_transacation = df_transacation.drop(cols_to_remove,axis=1)
df  = df_transacation.dropna()
y = df['isFraud']

X = df.drop(['isFraud'], axis=1)
categorical_feature_mask = X.dtypes=='category'

categorical_cols = X.columns[categorical_feature_mask].tolist()

categorical_cols
X_cat = X[categorical_cols]

X_cat = pd.get_dummies(X_cat)

X = X.drop(categorical_cols,axis=1)

X = pd.concat([X,X_cat], axis=1)
X.head()

X_cols = X.columns
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X)

X.head()
bestfeatures = SelectKBest(chi2, k=20)

fit = bestfeatures.fit(X,y)

# dfcolumns = pd.DataFrame(X_new.columns)

# dfcolumns

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_cols)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores = featureScores.sort_values(by='Score', ascending=False)

featureScores.head(20)
bestfeatures = SelectKBest(f_classif, k=20)

fit = bestfeatures.fit(X,y)

# dfcolumns = pd.DataFrame(X_new.columns)

# dfcolumns

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X_cols)

featureScores1 = pd.concat([dfcolumns,dfscores],axis=1)

featureScores1.columns = ['Specs','Score']  #naming the dataframe columns

featureScores1 = featureScores1.sort_values(by='Score', ascending=False)

featureScores1.head(20)
y = df['isFraud']

X = df.drop(['isFraud'], axis=1)
X1 = X[['V281', 'V308','V280','V279','V295','V293','V317']]



y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = train_test_split(X1, y['isFraud'], test_size=0.2)
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

# d_test = lgbm.Dataset(X_test, y_test)

params = {}

params['learning_rate'] = 0.02

params['boosting_type'] = 'gbdt'

# params['boosting_type'] = 'dart'

params['objective'] = 'binary'

params['metric'] = 'binary'

params['sub_feature'] = 0.99

params['num_leaves'] = 500

params['min_data'] = 100

params['max_depth'] = 10000

params['is_unbalance'] = True

# y_train=y_train.ravel()

clf = lgb.train(params, d_train, 1000)

from sklearn.metrics import f1_score, accuracy_score, fbeta_score

results=clf.predict(X_test)

score1 = fbeta_score(y_test,results.round(), beta=100 )

print('F1 Score ',score1)
import xgboost

xgb = xgboost.XGBClassifier(n_estimators=800, learning_rate=0.02, gamma=0, subsample=0.2,

                           colsample_bytree=1, max_depth=100)

xgb.fit(X_train,y_train)

results=xgb.predict(X_test)



score1 = fbeta_score(y_test,results.round(), beta=100 )

print('F Score - Recall:',score1)