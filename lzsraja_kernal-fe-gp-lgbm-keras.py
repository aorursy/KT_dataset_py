import numpy as np 

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt

import lightgbm as lgb

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras import backend as K

from keras import optimizers

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import  train_test_split

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import roc_auc_score

from gplearn.genetic import SymbolicTransformer
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

print('Training data shape:',data_train.shape)

print('Testing data shape:',data_test.shape)

data_train.head(5)
init_notebook_mode(connected=True)

labels = [str(x) for x in list(data_train['target'].unique())]

values = [(len(data_train[data_train['target'] == x])/len(data_train))*100 for x in list(data_train['target'].unique())]    

trace=go.Pie(labels=labels,values=values)

iplot([trace],filename = 'Target Percentages')
fe_train = data_train.copy()

fe_test = data_test.copy()

sd = True #standardize

st = True #statistics values

sp = False #sampling

gp = True #gplearning

if st:

    idx = features = fe_train.columns.values[2:202]

    for df in [fe_train, fe_test]:

        df['sum'] = df[idx].sum(axis=1)  

        df['min'] = df[idx].min(axis=1)

        df['max'] = df[idx].max(axis=1)

        df['mean'] = df[idx].mean(axis=1)

        df['std'] = df[idx].std(axis=1)

        df['skew'] = df[idx].skew(axis=1)

        df['kurt'] = df[idx].kurtosis(axis=1)

        df['med'] = df[idx].median(axis=1)

if sd:

    scaler = StandardScaler()

    fe_train.iloc[:,2:] = scaler.fit_transform(fe_train.iloc[:,2:])

    fe_test.iloc[:,1:] = scaler.fit_transform(fe_test.iloc[:,1:])

if gp:

    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg', 'max', 'min']

    generations = 80

    population_size = 2000

    gp = SymbolicTransformer(generations=generations, population_size=population_size,

                         hall_of_fame=100, n_components=10,

                         function_set=function_set,

                         parsimony_coefficient=0.0005,

                         max_samples=0.9, verbose=1,

                         random_state=0, n_jobs=3)

    gp.fit(fe_train.iloc[:,2:],fe_train['target'])

    gp_feature_name = ['NV'+str(i) for i in range(1, 11)]

    train_gp_features = pd.DataFrame(gp.transform(fe_train.iloc[:,2:]), columns=gp_feature_name)

    test_gp_features = pd.DataFrame(gp.transform(fe_test.iloc[:,1:]), columns=gp_feature_name)

    if sd:

        train_gp_features = pd.DataFrame(scaler.fit_transform(train_gp_features))

        test_gp_features = pd.DataFrame(scaler.fit_transform(test_gp_features))

    fe_train = pd.concat([fe_train, train_gp_features], axis=1)

    fe_test = pd.concat([fe_test, test_gp_features], axis=1)

if sp:

    fe_zeros = fe_train[fe_train['target'] == 0].sample(frac=0.25)

    fe_ones = fe_train[fe_train['target'] > 0]

    fe_train = pd.concat([fe_ones, fe_zeros]).sample(frac=1)

X = fe_train.iloc[:,2:]

Y = fe_train['target']

X_target = fe_test.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=113)

cv = True

param = {

    'bagging_freq': 8, #handling overfitting

    'bagging_fraction': 0.3,#handling overfitting - adding some noise

     #'boost': 'dart', 

    #'boost': 'goss',

     'boost_from_average':False,

     'boost': 'gbdt',   

    'feature_fraction': 0.2, #handling overfitting

    'learning_rate': 0.008, #the changes between one auc and a better one gets really small thus a small learning rate performs better

    'max_depth':2, 

    'metric':'auc',

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'xentropy', 

    'verbosity':1,

    "bagging_seed" : 122,

    "seed": 20,

    }

num_boost_round = 200000

verbose_eval=5000

if not cv:

    lgbm_train = lgb.Dataset(X_train,label=y_train)

    lgbm_test = lgb.Dataset(X_test,label=y_test)

    lgbm_valid = (lgbm_test,lgbm_train)

    valid_names = ['valid','train']

    lgbm_model = lgb.train(param,lgbm_train,num_boost_round=num_boost_round,valid_sets=lgbm_valid,valid_names=valid_names,

                          verbose_eval=verbose_eval,

                           early_stopping_rounds=5000)

    lgbm_predict = lgbm_model.predict(X_target)

else:

    lgbm_cv_valid_score = []

    lgbm_cv_predict = []

    kf = StratifiedKFold(n_splits=5,shuffle = False, random_state=311)

    for _fold, (trn_idx, val_idx) in enumerate(kf.split(X.values, Y.values)):

        print('Fold{}:'.format(_fold+1))

        X_cv_train, y_cv_train = X.iloc[trn_idx], Y.iloc[trn_idx]

        X_cv_test, y_cv_test = X.iloc[val_idx], Y.iloc[val_idx]

        lgbm_cv_train = lgb.Dataset(X_cv_train,label=y_cv_train)

        lgbm_cv_test = lgb.Dataset(X_cv_test,label=y_cv_test)

        lgbm_cv_valid = (lgbm_cv_test,lgbm_cv_train)

        valid_names = ['valid','train']

        lgbm_model = lgb.train(param,lgbm_cv_train,num_boost_round=num_boost_round,valid_sets=lgbm_cv_valid,valid_names=valid_names,

                          verbose_eval=verbose_eval,

                           early_stopping_rounds=5000)

        lgbm_cv_valid_score.append(roc_auc_score(y_cv_test,lgbm_model.predict(X_cv_test)))

        print('\tCVscore:{}'.format(lgbm_cv_valid_score[_fold]))

        print('-'*50)

        lgbm_cv_predict.append(lgbm_model.predict(X_target))

    print('---Mean CV score:{}---'.format(np.mean(lgbm_cv_valid_score)))

    lgbm_predict = np.mean(lgbm_cv_predict,axis = 0)
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc

input_dim  = X_train.shape[1]

epochs = 100

model = Sequential()

model.add(Dense(128,input_dim=input_dim, kernel_initializer='normal'))

model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001)

model.compile(optimizer=adam,loss='binary_crossentropy',metrics = [auc])

nn_model = model

nn_history = nn_model.fit(X_train, y_train, validation_data = (X_test ,y_test),

                          epochs=epochs,batch_size=64,verbose=1)

nn_predict = nn_model.predict(X_target)[:,0]
ensemble_predict3 = 0.1*nn_predict+0.9*lgbm_predict

ensemble_predict3[ensemble_predict3>1]=1

ensemble_predict3[ensemble_predict3<0]=0

submit = pd.DataFrame({'ID_code':fe_test['ID_code'],'target':ensemble_predict3})

submit.to_csv('nnlgbm_submission1.csv', index=False)

ensemble_predict4 = 0.3*nn_predict+0.7*lgbm_predict

ensemble_predict4[ensemble_predict4>1]=1

ensemble_predict4[ensemble_predict4<0]=0

submit = pd.DataFrame({'ID_code':fe_test['ID_code'],'target':ensemble_predict4})

submit.to_csv('nnlgbm_submission2.csv', index=False)


