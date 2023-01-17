import warnings

import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.feature_selection import SelectFromModel



warnings.filterwarnings('ignore')



seed = 7

np.random.seed(seed)



scale_const = 1
def load_data(path='../input/'):

    df_train = pd.read_csv(path.__add__('train.csv'))

    df_test = pd.read_csv(path.__add__('test.csv'))

    

    num_train = len(df_train)

    

    id_test = df_test['ID'].values

    

    y_train = df_train['y'].values.astype(np.float32)

    

    df_train_dummies = pd.get_dummies(df_train, drop_first=True)

    df_test_dummies = pd.get_dummies(df_test, drop_first=True)



    df_train_dummies = df_train_dummies.drop(['ID','y'], axis=1)

    df_test_dummies = df_test_dummies.drop('ID', axis=1)

    

    df_temp = pd.concat([df_train_dummies, df_test_dummies], join='inner')

    

    df_train = df_temp[:num_train]

    df_test = df_temp[num_train:]

    

    add_pca_ica_features(df_train, df_test)



    clf = ExtraTreesRegressor(n_estimators=300, max_depth=4, random_state=seed)



    clf.fit(df_train, y_train)



    features = pd.DataFrame()

    features['feature'] = df_train.columns

    features['importance'] = clf.feature_importances_

    features.sort_values(by=['importance'], ascending=True, inplace=True)

    features.set_index('feature', inplace=True)



    model = SelectFromModel(clf, prefit=True)

    train_reduced = model.transform(df_train)   



    test_reduced = model.transform(df_test.copy())

    

    df_train = pd.concat([df_train, pd.DataFrame(train_reduced)], axis=1)

    df_test = pd.concat([df_test, pd.DataFrame(test_reduced)], axis=1)

        

    df_all = pd.concat([df_train, df_test])

    

    x_train, x_test = df_all.values[:num_train], df_all.values[num_train:]  

                                   

    y_train /= scale_const

    

    return id_test, x_train, y_train, x_test

    

           

def add_pca_ica_features(train, test):    

    n_compute = 10



    # tSVD

    tsvd = TruncatedSVD(n_components=n_compute, random_state=seed)

    tsvd_results_train = tsvd.fit_transform(train)

    tsvd_results_test = tsvd.transform(test)



    # PCA

    pca = PCA(n_components=n_compute, random_state=seed)

    pca2_results_train = pca.fit_transform(train)

    pca2_results_test = pca.transform(test)



    # ICA

    ica = FastICA(n_components=n_compute, random_state=seed)

    ica2_results_train = ica.fit_transform(train)

    ica2_results_test = ica.transform(test)



    # GRP

    grp = GaussianRandomProjection(n_components=n_compute, eps=0.1, random_state=seed)

    grp_results_train = grp.fit_transform(train)

    grp_results_test = grp.transform(test)



# SRP

    srp = SparseRandomProjection(n_components=n_compute, dense_output=True, random_state=seed)

    srp_results_train = srp.fit_transform(train)

    srp_results_test = srp.transform(test)



#save columns list before adding the decomposition components



# Append decomposition components to datasets

    for i in range(1, n_compute + 1):

        train['pca_' + str(i)] = pca2_results_train[:, i - 1]

        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]

        test['ica_' + str(i)] = ica2_results_test[:, i - 1]



        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]



        train['grp_' + str(i)] = grp_results_train[:, i - 1]

        test['grp_' + str(i)] = grp_results_test[:, i - 1]



        train['srp_' + str(i)] = srp_results_train[:, i - 1]

        test['srp_' + str(i)] = srp_results_test[:, i - 1]





def inverse_scale(predict_value):  

    return scale_const * predict_value



id_test, x_train, y_train, x_test = load_data()
 # mmm, xgboost, loved by everyone ^-^

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 1000, 

    'eta': 0.001,

    'max_depth': 8,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': np.mean(y_train), # base prediction = mean(target)

    'silent': 1

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=500, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )



num_boost_rounds = len(cv_result)

print(num_boost_rounds)



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
from sklearn.metrics import r2_score



# now fixed, correct calculation

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results

y_pred = inverse_scale(model.predict(dtest))

output = pd.DataFrame({'id': id_test, 'y': y_pred.ravel()})

output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)