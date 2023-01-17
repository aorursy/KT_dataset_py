import numpy as np

import pandas as pd 

import os
#loading data 

os.listdir('../input')

sales_data = pd.read_csv('../input/sales_train.csv')

item_cat = pd.read_csv('../input/item_categories.csv')

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

test_data = pd.read_csv('../input/test.csv')

def basic_eda(df):

    print("----------TOP 5 RECORDS--------")

    print(df.head(5))

    print("----------INFO-----------------")

    print(df.info())

    print("----------Describe-------------")

    print(df.describe())

    print("----------Columns--------------")

    print(df.columns)

    print("----------Data Types-----------")

    print(df.dtypes)

    print("-------Missing Values----------")

    print(df.isnull().sum())

    print("-------NULL values-------------")

    print(df.isna().sum())

    print("-----Shape Of Data-------------")

    print(df.shape)

    

    
#Litle bit of exploration of data



print("=============================Sales Data=============================")

basic_eda(sales_data)

print("=============================Test data=============================")

basic_eda(test_data)

print("=============================Item Categories=============================")

basic_eda(item_cat)

print("=============================Items=============================")

basic_eda(items)

print("=============================Shops=============================")

basic_eda(shops)

print("=============================Sample Submission=============================")

basic_eda(sample_submission)



#we can see that 'date' column in sales_data is an object but if we want to manipulate 

#it or want to work on it someway then we have convert it on datetime format

sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')
#now we will create a pivot tabel by going so we get our data in desired form 

#we want get total count value of an item over the whole month for a shop 

# That why we made shop_id and item_id our indices and date_block_num our column 

# the value we want is item_cnt_day and used sum as aggregating function 

dataset = sales_data.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')

# lets reset our indices, so that data should be in way we can easily manipulate

dataset.reset_index(inplace = True)
# lets check on our pivot table

dataset.head()
# Now we will merge our pivot table with the test_data because we want to keep the data of items we have

# predict

dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')
# lets fill all NaN values with 0

dataset.fillna(0,inplace = True)

# lets check our data now 

dataset.head()
# we will drop shop_id and item_id because we do not need them

# we are teaching our model how to generate the next sequence 

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

# importing libraries required for our model

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
# our defining our model 

my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

my_model.add(Dropout(0.4))

my_model.add(Dense(1))



my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()

my_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)
# creating submission file 

submission_pfs = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)
