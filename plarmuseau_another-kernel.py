import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
from sklearn.decomposition import PCA, FastICA

 # mmm, xgboost, loved by everyone ^-^

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)



trainm=train

testm=test

stap=len(trainm)

plt.figure(figsize=(12,8))

sns.distplot(train['y'].values, bins=50, kde=False)

plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

plt.show()



for n_comp in range(4,44,10):

    del train,test

    train=trainm

    test=testm

    xtrain=train.drop(['y'],axis=1)

    print('PCA - ICA',n_comp)

    # PCA

    pca = PCA(n_components=n_comp, random_state=42)

    pca2_results_tot = pca.fit_transform(xtrain.append(test))

    # ICA

    ica = FastICA(n_components=n_comp, random_state=42)

    ica2_results_tot = ica.fit_transform(xtrain.append(test))

    # Append decomposition components to datasets

    for i in range(1, n_comp+1):

            train['pca_' + str(i)] = pca2_results_tot[:stap,i-1]

            test['pca_' + str(i)] = pca2_results_tot[stap:, i-1]

            train['ica_' + str(i)] = ica2_results_tot[:stap,i-1]

            test['ica_' + str(i)] = ica2_results_tot[stap:, i-1]

    #print(train.head(2))

    #print(test.head(2))

    y_train = train["y"]

    y_mean = np.mean(y_train)

    print(train.shape,test.shape)

    xgb_params = {

        'n_trees': 500, 

        'eta': 0.005,

        'max_depth': 4,

        'subsample': 0.95,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'base_score': y_mean, # base prediction = mean(target)

        'silent': 1

    }



    # form DMatrices for Xgboost training

    dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

    dtest = xgb.DMatrix(test)



    # xgboost, cross-validation

    cv_result = xgb.cv(xgb_params, 

                       dtrain, 

                       num_boost_round=700, # increase to have better results (~700)

                       early_stopping_rounds=100,

                       verbose_eval=50, 

                       show_stdv=False,

                       feval=xgb_r2_score

                    )



    num_boost_rounds = len(cv_result)

    print(num_boost_rounds)



    # train model

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    

    # check f2-score (to get higher score - increase num_boost_round in previous cell)



    print(r2_score(model.predict(dtrain), dtrain.get_label()))

    # make predictions and save results

    y_pred = model.predict(dtest)

    output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

    plt.figure(figsize=(12,8))

    sns.distplot(output.y.values, bins=50, kde=False)

    plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

    plt.show()

    output.to_csv('xgb-pca{}-ica.csv'.format(n_comp), index=False)