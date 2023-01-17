#Enivornment Setup
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read
file_path = '../input/car-price-prediction'
df=pd.read_csv(f'{file_path}/CarPrice_Assignment.csv')
df.head()
df = df.loc[:,~df.columns.duplicated()]
import pandas_profiling
# preparing profile report

profile_report = pandas_profiling.ProfileReport(df,minimal=True)
profile_report
df.info()
df.describe()
df.shape
df.price.value_counts()
df.apply(lambda x: sum(x.isnull()),axis=0)
df.groupby("CarName").mean()
def correlation_matrix(d):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Car Prediction dataset features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)
#Plotting data 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
sns.pairplot(df,diag_kind="kde")
plt.show()
drop_cols = ['car_ID'] 
df = df.drop(drop_cols, axis=1)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col])
df=df.copy()
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
new_train = df[df['price'].notnull()]
new_train
new_test = df[df['price'].isnull()].drop(['price'], axis=1)
new_test
X = new_train.drop('price', axis=1)
y = new_train['price']
#checking the target variable countplot
sns.countplot(data=new_train,x = 'price',palette='plasma')
from sklearn.model_selection import  train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape
X_train
# Distplot
fig, ax2 = plt.subplots(3, 3, figsize=(16, 16))
sns.distplot(df['horsepower'],ax=ax2[0][0])
sns.distplot(df['peakrpm'],ax=ax2[0][1])
sns.distplot(df['citympg'],ax=ax2[0][2])
sns.distplot(df['highwaympg'],ax=ax2[1][0])
sns.distplot(df['compressionratio'],ax=ax2[1][1])
sns.distplot(df['stroke'],ax=ax2[1][2])
sns.distplot(df['boreratio'],ax=ax2[2][0])
sns.distplot(df['boreratio'],ax=ax2[2][1])
sns.distplot(df['fuelsystem'],ax=ax2[2][2])
y_train
from sklearn.preprocessing import RobustScaler, StandardScaler
# Feature Scaling
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

target = "price"
def model(algorithm,dtrainx,dtrainy,dtestx,dtesty,of_type,plot=False):
    
    print (algorithm)
    print ("***************************************************************************")
    algorithm.fit(dtrainx,dtrainy)
    
    #print(algorithm.get_params(deep=True))
    
    prediction = algorithm.predict(dtestx)
    
    print ("ROOT MEAN SQUARED ERROR :", np.sqrt(mean_squared_error(dtesty,prediction)) )
    print ("***************************************************************************")
    
    print ('Performance on training data :', algorithm.score(dtrainx,dtrainy)*100)
    print ('Performance on testing data :', algorithm.score(dtestx,dtesty)*100)

    print ("***************************************************************************")
    if plot==True:
        sns.jointplot(x=dtesty, y=prediction, stat_func=pearsonr,kind="reg", color="b") 
    
       
    prediction = pd.DataFrame(prediction)
    cross_val = cross_val_score(algorithm,dtrainx,dtrainy,cv=5)#,scoring="neg_mean_squared_error"
    cross_val = cross_val.ravel()
    print ("CROSS VALIDATION SCORE")
    print ("************************")
    print ("cv-mean :",cross_val.mean()*100)
    print ("cv-std  :",cross_val.std()*100)
    
    if plot==True:
        plt.figure(figsize=(20,22))
        plt.subplot(211)

        testy = dtesty.reset_index()["price"]

        ax = testy.plot(label="originals",figsize=(20,9),linewidth=2)
        ax = prediction[0].plot(label = "predictions",figsize=(20,9),linewidth=2)
        plt.legend(loc="best")
        plt.title("ORIGINALS VS PREDICTIONS")
        plt.xlabel("index")
        plt.ylabel("values")
        ax.set_facecolor("k")
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
xgr =XGBRegressor(random_state=42)
model(xgr,X_train,y_train,X_test,y_test,"feat",True)
xgr_1=XGBRegressor(random_state=42,learning_rate = 0.03,
                max_depth = 9, n_estimators = 1000,n_jobs=-1,reg_alpha=0.005,gamma=0.1,subsample=0.7,colsample_bytree=0.9, colsample_bylevel=0.9, colsample_bynode=0.9)
model(xgr_1,X_train,y_train,X_test,y_test,"feat",True)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
param_grid={'n_estimators' : [1000,2000,3000,2500],
            'max_depth' : [1,2, 3,5,7,9,10,11,15],
            'learning_rate' :[ 0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.8, 1.0],
                                                     }
# Create a base model
xgbr = XGBRegressor(random_state = 42,reg_alpha=0.005,gamma=0.1,subsample=0.7,colsample_bytree=0.9, colsample_bylevel=0.9, colsample_bynode=0.9)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = xgbr, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
model(best_grid,X_train,y_train,X_test,y_test,"feat",True)
from sklearn.ensemble import  RandomForestRegressor
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
model(rf,X_train,y_train,X_test,y_test,"feat")
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=2000,
                             learning_rate=0.03,
                             depth=9,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
model(cb_model,X_train,y_train,X_test,y_test,"feat",True)
#Multiple Machine Learning Algorithm for Resgression 
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
model_dict = {
    'LinearRegession': LinearRegression(),
    'Ridge':Ridge(),
    'Lasso':Lasso(),
    'KernelRidge':KernelRidge(),
    'SGDRegressor':SGDRegressor(),
    'BayesianRidge':BayesianRidge(),
    'ElasticNet': ElasticNet(),
    'LinearSVR':LinearSVR(),
    #Perfect Models this Problem
    'XGBRegressor':XGBRegressor(random_state=42, n_estimators=2000, max_depth=9),
    'RandomForestRegressor': RandomForestRegressor(random_state=0, n_estimators=2000, max_depth=9),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42, n_estimators=2000, max_depth=9, learning_rate=0.01)
}
data_list = list()
for name, model in model_dict.items():
    data_dict = dict()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    data_dict['model'] = name
    data_dict['train_score'] = train_score
    data_dict['test_score'] = test_score
    data_list.append(data_dict)
score_df = pd.DataFrame(data_list)
score_df['score_diff'] = score_df['train_score'] - score_df['test_score']
model_df = score_df.sort_values(['test_score'], ascending=[False])
model_df[model_df['test_score'] > 0.5]
""""X = new_train.drop('price', axis=1)
y = new_train['price']
for ind, m_name in enumerate(model_df['model'].tolist()):
    model = model_dict[m_name].fit(X, y)
    predictions = model.predict(new_test)
    test['price'] = predictions
    test[['ID','price']].to_csv('Submission{}_{}.csv'.format(ind+1, m_name), index=False)"""
#Artificial Neural Networks(ANNs) Part:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Building ANN As a Regressor
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras import backend

#Defining Root Mean Square Error As our Metric Function 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# Initialising the Artificial Neural Networks(ANNs)
model_nn = Sequential()

# Adding the input layer and the first hidden layer
model_nn.add(Dense(512, activation = 'relu', input_dim = 24))
model_nn.add(BatchNormalization())
# Adding the second hidden layer
model_nn.add(Dense(units = 256, activation = 'relu'))
model_nn.add(BatchNormalization())
# Adding the third hidden layer
model_nn.add(Dense(units = 256, activation = 'relu'))
model_nn.add(BatchNormalization())
model_nn.add(Dense(units = 128, activation = 'relu'))
model_nn.add(BatchNormalization())
# Adding the output layer
model_nn.add(Dense(units = 1))

# Optimize , Compile And Train The Model 
opt =keras.optimizers.Adam(lr=0.003)
#print(model_nn.summary())
model_nn.compile(optimizer=opt,loss='mean_squared_error',metrics=[rmse])
import tensorflow as tf
checkpoint_filepath ='best.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_rmse',
    mode='min',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
history=model_nn.fit(sc.fit_transform(X_train),y_train,epochs = 300 ,batch_size=32,validation_data=(sc.transform(X_test), y_test), callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model_nn.load_weights(checkpoint_filepath)
# Predicting and Finding R Squared Score
y_predict = model_nn.predict(sc.transform(X_test))
print('Root Mean Squared Error is: ', np.sqrt(mean_squared_error(y_test, y_predict))) 

plt.figure(figsize=(20,5))
plt.plot(list(y_test) ,color = 'red', label = 'Real data',marker='o')
plt.plot(y_predict, color = 'blue', label = 'Predicted data',marker='o')
plt.title('Prediction')
plt.legend()
plt.show()

# Plotting Loss And Root Mean Square Error For both Training And Test Sets
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Root Mean Squared Error')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()