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
df=pd.read_csv('../input/energy-molecule/roboBohr.csv')
df.head()
#Unnamed: 0 adlı değişkeni silme:
#to clear which name is Unnamed: 0 columns
df=df.drop(['Unnamed: 0'],axis=1)
df.head()
#our dataset have 16242 observation units and 1277 columns
df.shape
#the dataset have nümeric variable(float64,int64),not catogarical columns(object,categorical)
df.info()
#have the dataset any null values in the observation? :No
df.isnull().sum().sum()
df.describe().T
#we add the independent variables to x variable
#we add the dependent variable to y variable
x=df.drop(['pubchem_id','Eat'],axis=1)
y=df['Eat']
#independent variables
x.head()
#dependent variable
y[0:10]
df.corr()
#now we will train-test split
#we will use function of train_test_split
from sklearn.model_selection import train_test_split
#train-test split:
#test_size:%80 train dataset,%20 test dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                              test_size=0.20,
                                              random_state=42)
#we look new size of train and test dataset:
print('x_train_shape:',x_train.shape)
print('x_test_shape:',x_test.shape)
print('y_train_shape:',y_train.shape)
print('y_test_shape',y_test.shape)
#now we will build a model
#we will use BaggingRegressor algorithm.
from sklearn.ensemble import BaggingRegressor
bag_model=BaggingRegressor(bootstrap_features=True) #object of model
bag_model.fit(x_train,y_train) #we fit the model and the model was builded
#we can learn hyperparameter which have model of bag_model(BaggedTrees algoritm)
# ?bag_model
#now we will learn how many trees have our model:
bag_model.n_estimators
#the model 10 trees,we will examine feautures of each trees:
bag_model.estimators_
#we will see how many observation units each trees
#this number is number of index the observation units
bag_model.estimators_samples_
bag_model.estimators_samples_[0].shape
bag_model.estimators_samples_[1].shape
#we will see independet values of each trees
bag_model.estimators_features_
bag_model.estimators_features_[0].shape
from sklearn.metrics import mean_squared_error,r2_score
y_pred=bag_model.predict(x_test)
#predicted values:
y_pred
#we will see primitive test error.
#this test error;bu test hatası,10 ağacın herbirinin verdiği tahminleri bir araya getirerek oluşturudğu tahminlerdir:
np.sqrt(mean_squared_error(y_test,y_pred))
#bağımsız değişkenlerin,bağımlı değişkeni açıklama başarısı:
r2_score(y_test,y_pred)
#2.ağaç:
iki_y_pred=bag_model.estimators_[1].fit(x_train,y_train).predict(x_test) #model kuruldu ve tahmin yapıldı
np.sqrt(mean_squared_error(y_test,iki_y_pred))
r2_score(y_test,iki_y_pred)
"""
#model nesnemizi yazalım:
bag_model=BaggingRegressor(bootstrap_features=True)
bag_model.fit(x_train,y_train)
"""
"""
#hiperparametre aralıkları:
bag_params={"n_estimators":range(2,5)}
"""
from sklearn.model_selection import GridSearchCV
"""
bag_cv_model=GridSearchCV(bag_model,
                         bag_params,
                         cv=10).fit(x_train,y_train)
"""
"""
#final model:
bag_tuned_model=BaggingRegressor(bootstrap_features=True,n_estimators=bag_cv_model.best_params).fit(x_train,y_train)
#tahmin
y_pred=bag_tuned_model.predict(x_test)
#test hatası:
np.sqrt(mean_squared_error(y_test,y_pred))
#r*2 skoru:
r2_score(y_test,y_pred)
"""

