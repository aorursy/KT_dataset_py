##importing toolkit

##basics

import pandas as pd

import numpy as np



##ploting

import matplotlib as plt



##balancing/ spliting

from imblearn.over_sampling import ADASYN

from sklearn.model_selection import train_test_split



##tunning & metrics

from skopt import dummy_minimize

from skopt import gp_minimize

from sklearn import metrics



##models

#import sklearn as skit

#from sklearn.svm import SVC

#from sklearn.tree import DecisionTreeClassifier

#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#from sklearn.naive_bayes import GaussianNB

import lightgbm as lgb



train_data = pd.read_csv(r'../input/santander-customer-transaction-prediction/train.csv')

test_data = pd.read_csv(r'../input/santander-customer-transaction-prediction/test.csv')
train_data.shape, test_data.shape
print('train data info:')

train_data.info() 

print('test data info:')

test_data.info()
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024 ** 2

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



    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df.head(5)
reduce_mem_usage(train_data)
train_data.head(10)
test_data.head(10)
train_data.describe()
test_data.describe()
def missing_values(df):

    total = df.isnull().sum()

    percent = (df.isnull().sum()/df.isnull().count()*100)

    tt = pd.concat([total, percent], axis = 1, keys = ['total', 'percent'])

    types = []

    for i in df.columns:

        dtype = str(df[i].dtype)

        types.append(dtype)

    tt['types'] = types

    

    if (df[i].isnull().sum() > 0):

        print('There is missing data')

        return (np.transpose(tt))

    else:

        print('There is no missing data')

        return (np.transpose(tt))
missing_values(train_data)
missing_values(test_data)
train_data['target'].hist(grid = False, figsize = (15, 7), color = 'black')
y = train_data.target

X = train_data.drop(columns = ['target', 'ID_code'])



# setting up testing and training sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify = y)



#sm = ADASYN()

#X_train, y_train = sm.fit_sample(X_train, y_train)
X.describe()
def train_model(params):

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    subsample = params[3]

    colsample_bytree = params[4]

    

    min_data_in_leaf = params[5]

    min_sum_hessian_in_leaf = params[6]

    bagging_fraction = params[7]

    bagging_freq = params[8]

    feature_fraction = params[9]

    lambda_l1 = params[10]

    lambda_l2 = params[11]

    min_gain_to_split = params[12]

    max_depth = params[13]

    verbosity = params[14]

    max_bin = params[15]

    

    

    print(params, '\n')

    

    mdl = lgb.LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=min_child_samples,

                        subsample=subsample, colsample_bytree=colsample_bytree, min_data_in_leaf = min_data_in_leaf, 

                        min_sum_hessian_in_leaf = min_sum_hessian_in_leaf, bagging_fraction = bagging_fraction,

                        bagging_freq = bagging_freq, feature_fraction = feature_fraction, lambda_l1 = lambda_l1, 

                        lambda_l2 = lambda_l2, min_gain_to_split = min_gain_to_split, max_depth = max_depth, 

                        verbosity = verbosity, max_bin = max_bin,

                        subsample_freq=1, n_estimators=100, n_jobs = -1, 

    objective = 'binary',

    metric = 'auc',

    boosting = 'gbdt')

    

    

    mdl.fit(X_train, y_train)

    

    p = mdl.predict_proba(X_val)[:,1]

    

    # Queremos minimizar o auc score

    return -metrics.roc_auc_score(y_val, p)



# Definindo nosso espaço de busca randômica. Não são tuplas, são ranges!

space = [(1e-3, 1e-1, 'log-uniform'), #learning rate

         (20, 80), # num_leaves

         (1, 100), # min_child_samples

         (0.05, 1.0), # subsample

         (0.1, 1.0), # colsample bytree

         (10, 20), #min_data_in_leaf

         (7, 14), #min_sum_hessian_in_leaf

         (0.5, 1), #bagging_fraction

         (1, 4), #bagging_freq

         (0.5, 1), #feature_fraction

         (0.3, 1.2), #lambda_l1

         (0.15, 0.6), #lambda_l2

         (0.01, 0.4), #min_gain_to_split

         (-2, 2), #max_depth

         (-2, 2), #verbosity

         (255, 500) #max_bin

        ] 



resultado = gp_minimize(train_model, space, verbose=1, n_calls=50, n_random_starts = 10, n_jobs = -1)
resultado.x
params = {

    'learning_rate': 0.05,

    'num_leaves': 70,

    'min_child_samples': 1,

    'subsample': 0.6279956574567085,

    'colsample_bytree': 1.0,

   

    'objective': 'binary',

    'metric': 'auc',

    'boosting': 'gbdt',

    'is_unbalanced': True,



    'min_data_in_leaf': 13,

    'min_sum_hessian_in_leaf': 14,

    'bagging_fraction': 0.5,

    'bagging_freq': 4,

    'feature_fraction': 0.7123317972623503,

    'lambda_l1': 0.37253228768374314,

    'lambda_l2': 0.45639156898526567,

    'min_gain_to_split': 0.1778400339886555,

    'max_depth': 2,

    'verbosity': 2,

    'max_bin': 346

}



print('Started training the model...')

##traing the LGBM with 2000 iterations and an early stop if model won't improove in 200 iterations.



mdl = lgb.LGBMClassifier(**params, n_estimators = 20000, n_jobs = -1)

#mdl = lgb.LGBMClassifier(**params, n_jobs = -1)

mdl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 

        verbose=1000, early_stopping_rounds= 200, eval_metric = 'roc_auc')

#mdl.fit(X_train, y_train)
name = mdl.__class__.__name__

print("="*30)

print(name)

print('****Results****')

p = mdl.predict_proba(X_val)[:, 1]

#print("Accuracy:", metrics.accuracy_score(y_val, p))

#print("Precision:", metrics.precision_score(y_val, p))

#print("Recall:", metrics.recall_score(y_val, p))

print("AUC:", metrics.roc_auc_score(y_val, p))
#test_to_sub = test_data.drop(columns=['ID_code'])

#print('test_to_sub ID Drop done.')

#print('Starting ID Separation...')

#ID = test_data['ID_code']

#print('ID Separation Done.')

#print('Started submission prediction...')

#Submission = mdl.predict_proba(test_to_sub)[:, 1]

#print('Prediction done.')

#print('Creating the .csv file...')

#CSV_to_Sub = pd.DataFrame({'ID_code': ID, 'target': Submission})

#print('Wrigting the .csv file...')

#CSV_to_Sub.to_csv(r'C:\Users\paulo\Desktop\customer_transaction_prediction\SUB6.csv', index = False)

#print(r'.csv file created at - C:\Users\paulo\Desktop\customer_transaction_prediction\SUB6.csv')
f = train_data.columns.values[2:202]

for df in [test_data, train_data]:

    df['sum'] = df[f].sum(axis=1)

    df['min'] = df[f].min(axis=1)

    df['max'] = df[f].max(axis=1)

    df['mean'] = df[f].mean(axis=1)

    df['std'] = df[f].std(axis=1)

    df['skew'] = df[f].skew(axis=1)

    df['kurt'] = df[f].kurtosis(axis=1)

    df['med'] = df[f].median(axis=1)
train_data[train_data.columns[202:]].head(5)
test_data[test_data.columns[202:]].head(5)
##balancing data with ADASYN technique

# Separate input features and target

y = train_data.target

X = train_data.drop(columns = ['target', 'ID_code'])



# setting up testing and training sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify = y)



#sm = ADASYN()

#X_train, y_train = sm.fit_sample(X_train, y_train)
X_train.head(2)
X_val.head(2)
def train_model(params):

    learning_rate = params[0]

    num_leaves = params[1]

    min_child_samples = params[2]

    subsample = params[3]

    colsample_bytree = params[4]

    

    min_data_in_leaf = params[5]

    min_sum_hessian_in_leaf = params[6]

    bagging_fraction = params[7]

    bagging_freq = params[8]

    feature_fraction = params[9]

    lambda_l1 = params[10]

    lambda_l2 = params[11]

    min_gain_to_split = params[12]

    max_depth = params[13]

    verbosity = params[14]

    max_bin = params[15]

    

    

    print(params, '\n')

    

    mdl = lgb.LGBMClassifier(num_leaves=num_leaves, min_child_samples=min_child_samples,

                        subsample=subsample, colsample_bytree=colsample_bytree, min_data_in_leaf = min_data_in_leaf, 

                        min_sum_hessian_in_leaf = min_sum_hessian_in_leaf, bagging_fraction = bagging_fraction,

                        bagging_freq = bagging_freq, feature_fraction = feature_fraction, lambda_l1 = lambda_l1, 

                        lambda_l2 = lambda_l2, min_gain_to_split = min_gain_to_split, max_depth = max_depth, 

                        verbosity = verbosity, max_bin = max_bin,

                        subsample_freq=1, n_estimators=100, learning_rate = learning_rate, n_jobs = -1, 

    objective = 'binary',

    metric = 'auc',

    boosting = 'gbdt')

    

    

    mdl.fit(X_train, y_train)

    

    p = mdl.predict_proba(X_val)[:,1]

    

    # Queremos minimizar o auc score

    return -metrics.roc_auc_score(y_val, p)



# Definindo nosso espaço de busca randômica. Não são tuplas, são ranges!

space = [(1e-3, 1e-1, 'log-uniform'), #learning rate

         (20, 80), # num_leaves

         (1, 100), # min_child_samples

         (0.05, 1.0), # subsample

         (0.1, 1.0), # colsample bytree

         (10, 20), #min_data_in_leaf

         (7, 14), #min_sum_hessian_in_leaf

         (0.5, 1), #bagging_fraction

         (1, 4), #bagging_freq

         (0.5, 1), #feature_fraction

         (0.3, 1.2), #lambda_l1

         (0.15, 0.6), #lambda_l2

         (0.01, 0.4), #min_gain_to_split

         (-2, 2), #max_depth

         (-2, 2), #verbosity

         (255, 500) #max_bin

        ] 



resultado = gp_minimize(train_model, space, verbose=1, n_calls=100, n_random_starts = 10, n_jobs = -1)
resultado.x
params = {

    'learning_rate': 0.01,

    'num_leaves': 80,

    'min_child_samples': 1,

    'subsample': 0.07014119574453273,

    'colsample_bytree': 0.30442008827557787,

   

    'objective': 'binary',

    'metric': 'auc',

    'boosting': 'gbdt',

    'is_unbalanced': True,

    'seed' : 0,



    'min_data_in_leaf': 17,

    'min_sum_hessian_in_leaf': 14,

    'bagging_fraction': 0.5,

    'bagging_freq': 1,

    'feature_fraction': 0.5,

    'lambda_l1': 1.2,

    'lambda_l2': 0.5493135486189045,

    'min_gain_to_split': 0.4,

    'max_depth': 2,

    'verbosity': 2,

    'max_bin': 500

}



print('Started training the model...')

##traing the LGBM with 2000 iterations and an early stop if model won't improove in 200 iterations.



mdl = lgb.LGBMClassifier(**params, n_estimators = 20000, n_jobs = -1)

#mdl = lgb.LGBMClassifier(**params, n_jobs = -1)

mdl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], 

        verbose=1000, early_stopping_rounds= 200, eval_metric = 'roc_auc')

#mdl.fit(X_train, y_train)



##changed seed = 0 // was none
name = mdl.__class__.__name__

print("="*30)

print(name)

print('****Results****')

p = mdl.predict_proba(X_val)[:, 1]

#print("Accuracy:", metrics.accuracy_score(y_val, p))

#print("Precision:", metrics.precision_score(y_val, p))

#print("Recall:", metrics.recall_score(y_val, p))

print("AUC:", metrics.roc_auc_score(y_val, p))
#test_to_sub = test_data.drop(columns=['ID_code'])

#print('test_to_sub ID Drop done.')

#print('Starting ID Separation...')

#ID = test_data['ID_code']

#print('ID Separation Done.')

#print('Started submission prediction...')

#Submission = mdl.predict_proba(test_to_sub)[:, 1]

#print('Prediction done.')

#print('Creating the .csv file...')

#CSV_to_Sub = pd.DataFrame({'ID_code': ID, 'target': Submission})

#print('Wrigting the .csv file...')

#CSV_to_Sub.to_csv(r'C:\Users\paulo\Desktop\customer_transaction_prediction\SUB_F_E2.csv', index = False)

#print(r'.csv file created at - C:\Users\paulo\Desktop\customer_transaction_prediction\SUB6_F_E2.csv')
##balancing data with ADASYN technique

# Separate input features and target

y = train_data.target

X = train_data.drop(columns = ['target', 'ID_code'])



# setting up testing and training sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify = y)



sm = ADASYN()

X_train, y_train = sm.fit_sample(X_train, y_train)
params = {'num_leaves': 128,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 16,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1.728910519108444,

         'reg_lambda': 4.9847051755586085,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 4}
model = lgb.LGBMClassifier(**params, n_estimators = 20000, n_jobs = -1)

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=1000, early_stopping_rounds=200)
sample_y_train = y_train.sample(frac=0.2)

sample_x_train = X_train.sample(frac=0.2)

sample_x_val = X_val.sample(frac=0.2)

sample_y_val = y_val.sample(frac=0.2)
X_train.shape, sample_x_train.shape
X_train.describe()
sample_x_train.describe()
sample_y_train.hist(grid = False, figsize = (15, 7), color = 'black')
# defining a list with all models 

classifiers = [

    #KNeighborsClassifier(3, n_jobs = -1),

    GaussianNB(),

    lgb.LGBMClassifier(n_jobs = -1),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_jobs = -1),

    GradientBoostingClassifier()]



# rotina para instanciar, predizer e medir os rasultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(sample_x_train, sample_y_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Results****')

    sample_y_pred = clf.predict(sample_x_val)

    print("Accuracy:", metrics.accuracy_score(sample_y_val, sample_y_pred))

    print("Precision:", metrics.precision_score(sample_y_val, sample_y_pred))

    print("Recall:", metrics.recall_score(sample_y_val, sample_y_pred))

    print("ROC:", metrics.roc_auc_score(sample_y_val, sample_y_pred))
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")