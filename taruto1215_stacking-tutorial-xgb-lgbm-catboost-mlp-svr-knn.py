import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import catboost

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import KFold



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trainval_filename = '/kaggle/input/restaurant-revenue-prediction/train.csv.zip'

test_filename = '/kaggle/input/restaurant-revenue-prediction/test.csv.zip'

df_trainval = pd.read_csv(trainval_filename)

df_test = pd.read_csv(test_filename)

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1,test_size=0.1)

print(df_trainval.head(5))
y_trainval = df_trainval['revenue']

del df_trainval['revenue']
print(df_trainval[['City Group','Type']].head())
df_all = pd.concat([df_trainval,df_test],axis=0)

df_all['Open Date'] = pd.to_datetime(df_all["Open Date"])

df_all['Year'] = df_all['Open Date'].apply(lambda x:x.year)

df_all['Month'] = df_all['Open Date'].apply(lambda x:x.month)

df_all['Day'] = df_all['Open Date'].apply(lambda x:x.day)

df_all['week_name'] = df_all['Open Date'].apply(lambda x:x.weekday_name)



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_all['City'] = le.fit_transform(df_all['City'])

df_all['City Group'] = df_all['City Group'].map({'Other':0,'Big Cities':1}) #There are only 'Other' or 'Big city'

df_all["Type"] = df_all["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3}) #There are only 'FC' or 'IL' or 'DT' or 'MB'

print(df_all.head())

df_all["week_name"] = df_all["week_name"].map({"Sunday":0, "Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}) #There are only 'FC' or 'IL' or 'DT' or 'MB'

print(df_all.head())
df_trainval = df_all.iloc[:df_trainval.shape[0]]

df_test = df_all.iloc[df_trainval.shape[0]:]
df_train_col = [col for col in df_trainval.columns if col not in ['Id','Open Date']]

df_trainval = df_trainval[df_train_col]

df_test = df_test[df_train_col]
import xgboost as xgb

class Model1Xgb:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        xgb_params = {'objective': 'reg:squarederror', #binary:logistic #multi:softprob,

                  'random_state': 10,

                  #'eval_metric': 'rmse'

                     }

        dtrain = xgb.DMatrix(tr_x, label=tr_y)

        dvalid = xgb.DMatrix(va_x, label=va_y)

        evals = [(dtrain, 'train'), (dvalid, 'eval')]

        self.model = xgb.train(xgb_params, dtrain, num_boost_round=10000,early_stopping_rounds=50, evals=evals)



    def predict(self, x):

        data = xgb.DMatrix(x)

        pred = self.model.predict(data)

        return pred
import lightgbm as lgb

class Model1lgb:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        lgb_params = {'objective': 'rmse',

                  'random_state': 10,

                  'metric': 'rmse'}

        lgb_train = lgb.Dataset(tr_x, label=tr_y)

        lgb_eval = lgb.Dataset(va_x, label=va_y,reference=lgb_train)

        self.model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval, num_boost_round=10000,early_stopping_rounds=50)



    def predict(self, x):

        pred = self.model.predict(x,num_iteration=self.model.best_iteration)

        return pred
import catboost

class Model1catboost:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        #https://catboost.ai/docs/concepts/python-reference_catboostregressor.html

        #catb = catboost.CatBoostClassifier(

        catb = catboost.CatBoostRegressor(

                                    iterations=10000, 

                                    use_best_model=True, 

                                    random_seed=10, 

                                    l2_leaf_reg=3,

                                    depth=6,

                                    loss_function="RMSE",#"CrossEntropy",

                                    #eval_metric = "RMSE", #'AUC',

                                    #classes_coun=3

                                  )

        self.model = catb.fit(tr_x,tr_y,eval_set=(va_x,va_y),early_stopping_rounds=50)

        print(self.model.score(va_x,va_y))

    def predict(self, x):

        pred = self.model.predict(x)

        return pred
from keras.models import Sequential

from keras.layers import Dense, Dropout



from keras.callbacks import EarlyStopping



class Model1NN:



    def __init__(self):

        self.model = None

        self.scaler = None

    '''

    def weight_variable(self,shape,name):

        initial =tf.truncated_normal(shape,stddev=0.1)

        return tf.Variable(initial, name=name)



    def bias_variable(self,shape,name):

        initial = tf.constant(0.1,shape=shape)

        return tf.Variable(initial, name=name)

    '''    

    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        

        batch_size = 128

        epochs = 10000

        

        tr_x = self.scaler.transform(tr_x)

        va_x = self.scaler.transform(va_x)

        

        early_stopping =  EarlyStopping(

                            monitor='val_loss',

                            min_delta=0.0,

                            patience=20,

        )



        model = Sequential()

        model.add(Dense(32, activation='relu', input_shape=(tr_x.shape[1],)))

        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))

        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))



        model.compile(loss='mean_squared_error', #'categorical_crossentropy',#categorical_crossentropy

                      optimizer='adam')



        history = model.fit(tr_x, tr_y,

                            batch_size=batch_size, epochs=epochs,

                            verbose=1,

                            validation_data=(va_x, va_y),

                            callbacks=[early_stopping])

        self.model = model



    def predict(self, x):

        x = self.scaler.transform(x)

        pred = self.model.predict_proba(x).reshape(-1)

        return pred
from sklearn.svm import LinearSVR



class Model1LinearSVR:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        #params = {"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

        self.model = LinearSVR(max_iter=1000,

                               random_state=10,

                               C = 1.0, #損失の係数（正則化係数の逆数）

                               epsilon = 5.0

                               

                              )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.svm import SVR



class Model1KernelSVR:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        #params = {"kernel":['rbf'],"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

        self.model = SVR(kernel='rbf',

                         gamma='auto',

                         max_iter=1000,

                         C = 1.0, #損失の係数（正則化係数の逆数）

                         epsilon = 5.0

                         

                              )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.linear_model import Lasso



class Model1Lasso:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        self.model = Lasso(

            alpha=1, #L1係数

            fit_intercept=True,

            )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.linear_model import Ridge



class Model1Ridge:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        self.model = Ridge(

                            alpha=1, #L2係数

                              )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.linear_model import ElasticNet



class Model1ElasticNet:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        '''1 / (2 * n_samples) * ||y - Xw||^2_2

        + alpha * l1_ratio * ||w||_1

        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

       ref)  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

        '''

        

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        self.model = ElasticNet(

            alpha=1, #L1係数

            l1_ratio=0.5,

                              )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.ensemble import RandomForestRegressor



class Model1RF:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        self.model = RandomForestRegressor(

            max_depth=5,

            n_estimators=100,

            random_state=10,

        )

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.neighbors import KNeighborsRegressor



class Model1KNN:



    def __init__(self):

        self.model = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        #params = {"kernel":['rbf'],"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

        self.model = KNeighborsRegressor(n_neighbors=5,

                                         #weights='uniform'

                                        )

        

        self.model.fit(tr_x,tr_y)

        

    def predict(self,x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
from sklearn.linear_model import LinearRegression



class Model2Linear:



    def __init__(self):

        self.model = None

        self.scaler = None



    def fit(self, tr_x, tr_y, va_x, va_y):

        self.scaler = StandardScaler()

        self.scaler.fit(tr_x)

        tr_x = self.scaler.transform(tr_x)

        self.model = LinearRegression()

        self.model.fit(tr_x, tr_y)



    def predict(self, x):

        x = self.scaler.transform(x)

        pred = self.model.predict(x)

        return pred
def predict_cv(model, train_x, train_y, test_x):

    preds = []

    preds_test = []

    va_idxes = []



    kf = KFold(n_splits=4, shuffle=True, random_state=10)



    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model.fit(tr_x, tr_y, va_x, va_y)

        pred = model.predict(va_x)

        preds.append(pred)

        pred_test = model.predict(test_x)

        preds_test.append(pred_test)

        va_idxes.append(va_idx)



    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す

    va_idxes = np.concatenate(va_idxes)

    preds = np.concatenate(preds, axis=0)

    order = np.argsort(va_idxes)

    pred_train = preds[order]



    # テストデータに対する予測値の平均をとる

    preds_test = np.mean(preds_test, axis=0)



    return pred_train, preds_test
model_1a = Model1Xgb()

pred_train_1a, pred_test_1a = predict_cv(model_1a, df_trainval, y_trainval, df_test)
model_1b = Model1lgb()

pred_train_1b, pred_test_1b = predict_cv(model_1b, df_trainval, y_trainval, df_test)
model_1c = Model1NN()

pred_train_1c, pred_test_1c = predict_cv(model_1c, df_trainval, y_trainval, df_test)
model_1d = Model1LinearSVR()

pred_train_1d, pred_test_1d = predict_cv(model_1d, df_trainval, y_trainval, df_test)
model_1e = Model1KernelSVR()

pred_train_1e, pred_test_1e = predict_cv(model_1e, df_trainval, y_trainval, df_test)
model_1f = Model1catboost()

pred_train_1f, pred_test_1f = predict_cv(model_1f, df_trainval, y_trainval, df_test)
model_1g = Model1KNN()

pred_train_1g, pred_test_1g = predict_cv(model_1g, df_trainval, y_trainval, df_test)
model_1h = Model1Lasso()

pred_train_1h, pred_test_1h = predict_cv(model_1h, df_trainval, y_trainval, df_test)
model_1i = Model1Ridge()

pred_train_1i, pred_test_1i = predict_cv(model_1i, df_trainval, y_trainval, df_test)
model_1j = Model1ElasticNet()

pred_train_1j, pred_test_1j = predict_cv(model_1j, df_trainval, y_trainval, df_test)
model_1k = Model1RF()

pred_train_1k, pred_test_1k = predict_cv(model_1k, df_trainval, y_trainval, df_test)
from sklearn.metrics import mean_absolute_error



print(f'a LGBM mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1a):.4f}')

print(f'b XGBoostmean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1b):.4f}')

print(f'c MLP mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1c):.4f}')

print(f'd LinearSVR mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1d):.4f}')

print(f'e KernelSVR mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1e):.4f}')

print(f'f Catboost mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1f):.4f}')

print(f'g KNN mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1g):.4f}')

print(f'h Lasso mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1h):.4f}')

print(f'i Ridge mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1i):.4f}')

print(f'j ElasticNet mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1j):.4f}')

print(f'k RandomForest mean_absolute_error: {mean_absolute_error(y_trainval,pred_train_1k):.4f}')
train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a,

                          'pred_1b': pred_train_1b,

                          'pred_1c': pred_train_1c,

                          #'pred_1d': pred_train_1d,

                          'pred_1e': pred_train_1e,

                          'pred_1f': pred_train_1f,

                          'pred_1g': pred_train_1g,

                          'pred_1h': pred_train_1h,

                          'pred_1i': pred_train_1i,

                          'pred_1j': pred_train_1j,

                          'pred_1k': pred_train_1k,

                         })

test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a,

                          'pred_1b': pred_test_1b,

                          'pred_1c': pred_test_1c,

                          #'pred_1d': pred_test_1d,

                          'pred_1e': pred_test_1e,

                          'pred_1f': pred_test_1f,

                          'pred_1g': pred_test_1g,

                          'pred_1h': pred_test_1h,

                          'pred_1i': pred_test_1i,

                          'pred_1j': pred_test_1j,

                          'pred_1k': pred_test_1k

                         })
model2 = Model2Linear()

pred_train_2, pred_test_2 = predict_cv(model2, train_x_2, y_trainval, test_x_2)
print(f'mean_absolute_error: {mean_absolute_error(y_trainval, pred_train_2):.4f}')

print('best a,b,c,e,f,g,h,i,j,k')
df_test = pd.read_csv(test_filename)
submission = pd.DataFrame({'Id':df_test['Id'],'Prediction':pred_test_2})
submission.to_csv('./submission200214_fold４_10models.csv',index=False)

!ls && pwd