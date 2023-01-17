import os

import datetime

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='darkgrid')

pd.set_option('display.float_format',lambda  x: '%.2f' %x)

sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
print('sales:',sales.shape,'test:',test.shape,'items:',items.shape,'item_cats:',item_cats.shape,'shop:',shops.shape)
sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales.head(3)

test.head(3)
test.shape

items.head(3)
item_cats.head(3)
shops.head(3)
sales[sales['item_price']<=0]
sales[(sales.shop_id==32)&(sales.item_id==2973)&(sales.date_block_num==4)]
median = sales[(sales.shop_id==32)&(sales.item_id==2973)&(sales.date_block_num==4)&

               (sales.item_price>0)].item_price.median()
sales.loc[sales.item_price<0,'item_price'] =median


dataset = sales.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')
dataset


dataset.reset_index(inplace = True)
dataset.head()
dataset.shape
dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')
dataset
dataset.fillna(0,inplace = True)



dataset.head()
dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset.head()
# X we will keep all columns execpt the last one 

X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# the last column is our label

y_train = dataset.values[:,-1:]



# for test we keep all the columns execpt the first one

X_test = np.expand_dims(dataset.values[:,1:],axis = 2)



# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
# importing libraries 

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

my_model.add(Dropout(0.3)) #The dropout rate is set to 30%, meaning one in 3.33 inputs will be randomly excluded from each update cycle.

my_model.add(Dense(1))



my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()
lst_pred = my_model.fit(X_train,y_train,batch_size = 4000,epochs = 8)
y_pred = my_model.predict(X_test)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):

    grid_model = Sequential()

    grid_model.add(LSTM(units = 64,input_shape = (33,1)))

    grid_model.add(Dropout(0.4))

    grid_model.add(Dense(1))



    grid_model.compile(loss = 'mse',optimizer = optimizer, metrics = ['mean_squared_error'])

    return my_model



grid_model = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size' : [4000,4080],

              'epochs' : [8,10],

              'optimizer' : ['adam','Adadelta'] }



grid_search  = GridSearchCV(estimator = grid_model,

                            param_grid = parameters,

                            scoring = 'accuracy',

                            cv = 2)

grid_search = grid_search.fit(X_train,y_train)


best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
# Let us check our best parameters

best_parameters
# Let us check our best accuracy got through grid search

best_accuracy
plt.plot(lst_pred.history['loss'], label= 'loss(mse)')

plt.plot(np.sqrt(lst_pred.history['mean_squared_error']), label= 'rmse')

plt.legend(loc=1)
from sklearn.metrics import mean_squared_error

import math
# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(y_train, y_pred[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))
# creating submission file 

submission_pfs = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)