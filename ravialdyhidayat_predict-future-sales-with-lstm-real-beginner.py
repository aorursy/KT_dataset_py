#Import packages that are needed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

df_test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
#Check sales_train data

sales_train.head(2)
#Check items data

items.head(2)
#Check item_categories

item_categories.head(2)
#Check shops

shops.head(2)
#Join multiple separate training data to one dataframe

df_train = sales_train.join(items, rsuffix = "*").join(shops, rsuffix = "*").join(item_categories, rsuffix = "*").drop("item_name", axis = 1)

df_train.head(3)
#Check test data

df_test.head(3)
#Select the training data that is belonged to testing data

df_train = df_train[df_train["shop_id"].isin(df_test["shop_id"]) & df_train["item_id"].isin(df_test["item_id"])]

df_train.head(3)
#Choose the most informative features and aggregate with respect to 'item_cnt_day'

df_train0 = df_train[['date', 'date_block_num', 'shop_id', 'item_id', 

                      'item_cnt_day']].sort_values('date').groupby(['date_block_num', 

                                                                    'shop_id', 'item_id'], as_index=False).agg({

    'item_cnt_day': ['sum']

})

df_train0.columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']

df_train0.head(3)
#Check any missing value

df_train0.isna().sum()
df_train1 = df_train0.pivot_table(columns = 'date_block_num', index = ['shop_id', 'item_id'], values = 'item_cnt_day', fill_value = 0, 

                                 aggfunc = 'sum').reset_index()

df_train1.head(3)
df_data = pd.merge(df_test, df_train1, how = 'left', on = ['item_id','shop_id'])

df_data.head(3)
#Fill all missing values with 0

df_data = df_data.fillna(0)

df_data.head(3)
#drop uninformative features

df_data.drop(['ID', 'shop_id', 'item_id'], axis = 1, inplace = True)

df_data.head(3)
# Create data training and testing

X_train = np.expand_dims(df_data.values[:,:-1],axis = 2)

X_test = np.expand_dims(df_data.values[:,1:],axis = 2)



# the last column/feature is our target variable

y_train = df_data.values[:,-1:]



print(X_train.shape,y_train.shape,X_test.shape)
#Import packages that are fundamental to our model.

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
modelML = Sequential()

modelML.add(LSTM(units = 64,input_shape = (33,1)))

modelML.add(Dropout(0.4))

modelML.add(Dense(1))



modelML.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

modelML.summary()
modelML.fit(X_train,y_train,batch_size = 4096,epochs = 10)
# Predict and Submit

submission = modelML.predict(X_test)



submission = pd.DataFrame({

    'ID': df_test['ID'],

    'item_cnt_month': submission.ravel()

})

# create csv file

submission.to_csv(r'E:\Elimination_Round_DAC_2020\submission_predict_sales.csv',index = False)