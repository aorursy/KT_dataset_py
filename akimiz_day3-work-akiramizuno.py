%matplotlib inline

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import mean_absolute_error 



#小数点がカンマで表記されているため、decimalパラメータを付与

df = pd.read_csv("../input/measurements.csv",decimal=',')

df.head()
#欠損値の確認

df.info()
# 欠損値は欠損していないデータの平均とする

tmp_temp_inside = df["temp_inside"].dropna()

tmp_temp_inside = tmp_temp_inside.astype(np.float)

tmp_temp_inside_avg = tmp_temp_inside.mean()

df["temp_inside"] = tmp_temp_inside

df["temp_inside"] = df["temp_inside"].fillna(tmp_temp_inside_avg)
#変数gas_typeをone hot encoding

def distinctGasType(x):

    if x == "E10":

        return 0

    else:

        return 1



df["gas_type_ohe"] = df["gas_type"].apply(lambda x: distinctGasType(x))
pd.plotting.scatter_matrix(df, figsize=(10,10))

plt.show()
sns.heatmap(df.corr(),cmap='coolwarm',vmin=-1,vmax=1)

plt.show()
from sklearn.model_selection import train_test_split

X = df.drop(columns=['consume','gas_type','specials','refill liters','refill gas']).values

y = df['consume'].values.reshape(-1,1)

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_norm = X_train

X_test_norm = X_test

X_train_norm[:,:4] = stdsc.fit_transform(X_train[:,:4])

X_test_norm[:,:4] = stdsc.transform(X_test[:,:4])
#パラメータ決定

#test_size = 0.2

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)



#線形回帰

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train_norm, y_train)

y_pred = regr.predict(X_test_norm)



df_graph = pd.DataFrame()

df_graph['class'] = ['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']

df_graph['coef']  = regr.coef_.ravel()

sns.set()

sns.catplot(data=df_graph,x='class',y='coef',kind='bar',height=3, aspect=2.5)
mae = mean_absolute_error(y_test, y_pred) 

print("MAE = %s"%round(mae,3) )
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import graphviz

#import pydotplus

from IPython.display import Image

from sklearn.externals.six import StringIO

from sklearn.tree import DecisionTreeRegressor, export_graphviz
#X_train = df[["x1","x2"]].values

#y_train = df["label"].values

clf = DecisionTreeRegressor(max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)

clf = clf.fit(X_train_norm, y_train)

print("score=", clf.score(X_train_norm, y_train))

#print(clf.predict(X_test)) #予測したい場合

y_pred = clf.predict(X_test_norm)

mae = mean_absolute_error(y_test, y_pred) 

print("MAE = %s"%round(mae,3) )
# 説明変数の重要度を出力する

# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。

print(clf.feature_importances_)

pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()
count=0

for y_p in y_pred:

    if y_p in y_train:

        count+=1

print("count = %d"%count)

print("size of y_pred = %d"%len(y_pred))
from sklearn.ensemble import RandomForestRegressor



clf = RandomForestRegressor(n_estimators=10, max_depth=2, 

                            min_samples_leaf=2, min_samples_split=2, random_state=1234)

clf.fit(X_train_norm, y_train)

print("score=", clf.score(X_train_norm, y_train))



# 説明変数の重要度を出力する

# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。

print(clf.feature_importances_)

pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()

mae = mean_absolute_error(y_test, y_pred) 

print("MAE = %s"%round(mae,3) )
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor



clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3,min_samples_leaf=2,min_samples_split=2,random_state=1234),

                                           n_estimators=10, random_state=1234)

clf.fit(X_train_norm, y_train)

print("score=", clf.score(X_train_norm, y_train))



# 説明変数の重要度を出力する

# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。

print(clf.feature_importances_)

pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))

plt.ylabel("Importance")

plt.xlabel("Features")

plt.show()

mae = mean_absolute_error(y_test, y_pred) 

print("MAE = %s"%round(mae,3) )
import tensorflow as tf

from tensorflow import keras 

print(tf.__version__)



# import libraries

import numpy as np

import pandas as pds

from keras.models import Sequential

from keras.layers import Input, Dense, Dropout, BatchNormalization

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_diabetes
# create regression model

def reg_model(optimizer='rmsprop', init='glorot_uniform'):

    model = Sequential()

    model.add(Dense(10, input_dim=8, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1))

    # compile model

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



# show the model summary

reg_model().summary()
%%time



# KerasClassifier/KerasRegressor can be used as same as scikit_learn estimator.

estimator = KerasRegressor(build_fn=reg_model, verbose=0)



# Grid Search parameters (epochs, batch size and optimizer)

#optimizers = ['rmsprop', 'adam']

#init = ['glorot_uniform', 'normal', 'uniform']

#epochs = [10, 20, 30]

#batches = [5, 10, 20]

#optimizers = ['rmsprop']

#init = ['normal']

#epochs = [30]

#batches = [5]

#param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

#clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=4, scoring="r2", return_train_score=True, iid=False)

#clf.fit(X_train_norm, y_train.ravel())

#print(clf.best_params_)



# 最適パラメータを用いて識別する

clf2 = KerasRegressor(build_fn=reg_model,verbose=0)

clf2.fit(X_train_norm, y_train.ravel())

y_pred = clf.predict(X_test_norm)
mae = mean_absolute_error(y_test, y_pred) 

print("MAE = %s"%round(mae,3) )