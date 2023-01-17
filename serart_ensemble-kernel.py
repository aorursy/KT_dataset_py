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



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def add_features(train, test):    

    n_compute = 20



    tsvd = TruncatedSVD(n_components=n_compute, random_state=seed)

    tsvd_results_train = tsvd.fit_transform(train)

    tsvd_results_test = tsvd.transform(test)



    pca = PCA(n_components=n_compute, random_state=seed)

    pca2_results_train = pca.fit_transform(train)

    pca2_results_test = pca.transform(test)



    ica = FastICA(n_components=n_compute, random_state=seed)

    ica2_results_train = ica.fit_transform(train)

    ica2_results_test = ica.transform(test)

    

    grp = GaussianRandomProjection(n_components=n_compute, eps=0.1, random_state=seed)

    grp_results_train = grp.fit_transform(train)

    grp_results_test = grp.transform(test)



    srp = SparseRandomProjection(n_components=n_compute, dense_output=True, random_state=seed)

    srp_results_train = srp.fit_transform(train)

    srp_results_test = srp.transform(test)



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
scale_const = 1



def inverse_scale(predict_value):  

    return scale_const * predict_value
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

    

    add_features(df_train, df_test)



    clf = ExtraTreesRegressor(n_estimators=250, max_depth=4, random_state=seed)



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
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)



class Ensemble(object):

    def __init__(self, stack, weights='mean', bias=None):

        self.stack = stack

        self.weights = weights

        self.bias = bias

        self.x = None

        

    def fit(self, x_train, y_train, lr_moe=0.001, n_epochs_moe=10):

        len_s = len(self.stack)

        

        for b in self.stack:

            b.fit(x_train, y_train)

            print('\n')

        

        if self.weights == 'mean':

            self.weights = np.full(len_s, 1./len_s)

        elif self.weights == 'moe':

            print('Train moe algorithm on {}\n'.format(x_train.shape[0]))

            

            self.x = np.random.uniform(low=0, high=1, size=len_s)

            self.bias = np.random.uniform(low=0, high=5, size=len_s)

            

            for t in range(1, n_epochs_moe + 1):                

                y_predict = np.vstack(self.stack_predict(x_train))

                prob = softmax(self.x)        

                

                for k in range(x_train.shape[0]):

                    dG_dx = []

                    dG_db = []

                    

                    for i in range(len_s):                     

                        dG_dx.append(0.5 * prob[i] * (1 - prob[i]) * (y_train - y_predict[i][k] + self.bias[i])**2)

                        dG_db.append(prob[i] * (y_train - y_predict[i][k] + self.bias[i]))

                             

                        self.x[i] -= lr_moe * dG_dx[i][k]

                        self.bias[i] -= lr_moe * dG_db[i][k]

                    

                print('Epoch {}; weights {}; alpha {}'.format(t, softmax(self.x), self.bias))

                

            self.weights = softmax(self.x)

        

    def stack_predict(self, x_valid):

        b = [] 

        for b_ in self.stack:

            b.append(b_.predict(x_valid))

            

        return b

    

    def predict(self, x_valid):              

        predict = np.average(self.stack_predict(x_valid), axis=0, weights=self.weights)

        

        if self.bias != None:

            predict += np.average(self.bias, weights=self.weights, axis=0)

        return predict

  



print('Ensemble class succsessful build')
def r2_score_keras(y_true, y_pred):

    SS_res = K.sum(K.square(y_true - y_pred)) 

    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    

    return (1 - SS_res / (SS_tot + K.epsilon()))
from keras import backend as K

from keras.models import Sequential, load_model

from keras.layers import Dense, LSTM, Dropout, Reshape, Activation, BatchNormalization

from keras.optimizers import sgd



from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



from sklearn.linear_model import LassoLarsCV



import xgboost as xgb



id_test, x_train, y_train, x_test = load_data()



input_shape = (x_test.shape[1], )

re_input_shape = (1, x_test.shape[1]) 



x_train, x_valid, y_train, y_valid = train_test_split(

    x_train, 

    y_train, 

    test_size=0.2, 

    random_state=seed

)
def nn_model():       

    model = Sequential()

    

    model.add(Reshape(re_input_shape, input_shape=input_shape))

    model.add(LSTM(100, return_sequences=False))

    model.add(Dropout(0.2))

    

    model.add(Dense(250))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    

    model.add(Dense(1, activation='linear'))

    

    model.compile(optimizer='adam',

                  loss='mse', 

                  metrics=[r2_score_keras])

   

    return model



nn_regressor = KerasRegressor(build_fn=nn_model, 

                           epochs=150,  

                           batch_size=32,

                           validation_data=(x_valid, y_valid), 

                           shuffle=True,

                           verbose=2)



gbr = GradientBoostingRegressor(learning_rate=0.001, 

                                loss="huber", 

                                max_depth=3, 

                                max_features=0.55, 

                                min_samples_leaf=18, 

                                min_samples_split=14, 

                                subsample=0.7)



lasso = LassoLarsCV(normalize=True)



rfr = RandomForestRegressor(n_estimators=250, 

                           min_samples_leaf=25,

                           min_samples_split=25,

                           n_jobs=4,

                           max_depth=4)
# Xgboost

import xgboost as xgb



xgb_params = {

    'n_trees': 1500, 

    'eta': 0.006,

    'max_depth': 6,

    'subsample': 0.93,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': np.mean(y_train),

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)

dvalid = xgb.DMatrix(x_valid)



model_xgb = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=750)
model = Ensemble(stack=[nn_regressor, gbr, rfr, lasso], weights='mean')

model.fit(x_train, y_train)



print('Done')
y_predict = 0.25 * inverse_scale(model.predict(x_test)) + 0.75 * inverse_scale(model_xgb.predict(dtest)) 

y_predict_test = 0.25 * inverse_scale(model.predict(x_train)) + 0.75 * inverse_scale(model_xgb.predict(dtrain))

y_predict_valid = 0.25 * inverse_scale(model.predict(x_valid)) + 0.75 * inverse_scale(model_xgb.predict(dvalid))





df = pd.DataFrame({'ID':id_test, 'y':y_predict.ravel()})

df.to_csv('stack_test.csv', index=False)



print('R2 score test', r2_score(inverse_scale(y_train), y_predict_test))

print('R2 score test', r2_score(inverse_scale(y_valid), y_predict_valid))