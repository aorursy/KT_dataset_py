import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rcParams 
%matplotlib inline 
rcParams['figure.figsize']=10,8
sns. set (style='whitegrid' , palette='muted' , rc={'figure.figsize' :(15,10)})
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential 
from keras.layers import Dense , Activation , Dropout
from numpy.random import seed 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
train.head()
train['ocean_proximity']=train['ocean_proximity'].astype('category')
train['ocean_proximity']=train['ocean_proximity'].cat.codes
continuous=['longitude' ,'latitude' , 'housing_median_age' , 'total_rooms' , 'total_bedrooms','population','households','median_income' ,'median_house_value' ]
scaler=MinMaxScaler()
for var in continuous :
    train[var]=train[var].astype('float64')
    train[var]=scaler.fit_transform(train[var].values.reshape(-1,1))
train.head()
e8=0.8*(len(train))
display("e8="+(str)(e8))
x_train=train[pd.notnull(train['total_bedrooms'])].drop(['median_house_value'],axis=1)[0:(int)(e8)]
y_train=train[pd.notnull(train['total_bedrooms'])]['median_house_value'][0:(int)(e8)]
x_test=train[pd.notnull(train['total_bedrooms'])].drop(['median_house_value'],axis=1)[(int)(e8):len(train)]
y_test=train[pd.notnull(train['total_bedrooms'])]['median_house_value'][(int)(e8):len(train)]


display("x train ="+(str)(len(x_train))+"y train="+(str)(len(y_train))+"x test="+(str)(len(x_test))+"y test="+(str)(len(y_test)))
x_train.head()
y_train.head()
def create_model(lyrs=[8] , act='relu' , opt='Adam' , dr=0.0):
    model=Sequential()
    model.add(Dense(lyrs[0], input_dim=x_train.shape[1],activation=act))
    for i in range(1,len(lyrs)):
        model.add(dense(lyrs[i] , activation=act))
    model.add(Dropout(dr))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error' , optimizer=opt , metrics=['accuracy'])
    return model

model=create_model()
print(model.summary())
training=model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.2,verbose=0 )
val_acc=np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%"%('val_accuracy',(val_acc*100)))
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()


model=KerasClassifier(build_fn=create_model,verbose=0)
batch_size=[16,32,64]
epochs=[50,100]
param_grid=dict(batch_size=batch_size,epochs=epochs)
grid=GridSearchCV(estimator=model , param_grid=param_grid , cv=3,verbose=0)
grid_result=grid.fit(x_train,y_train)
print("%f using %s" % (grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params =grid_result.cv_results_['params']
for mean,stddev,param in zip(means,stds,params):
        print("%f(%f)with: %r"% (mean,stddev,param))