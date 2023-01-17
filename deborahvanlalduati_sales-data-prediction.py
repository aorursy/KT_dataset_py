import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



path = "/kaggle/input/competitive-data-science-predict-future-sales/"



items = pd.read_csv(path+'/items.csv')

item_cats = pd.read_csv(path+'/item_categories.csv')

shops = pd.read_csv(path+'/shops.csv')

sales = pd.read_csv(path+'/sales_train.csv')

test = pd.read_csv(path+'/test.csv')

submission = pd.read_csv(path+'/sample_submission.csv')



print("Data set loaded successfully.")

print(items.info())

print('Items : \n\t'+'\n\t'.join(list(items)))

print('ItemsCatagories : \n\t'+'\n\t'.join(list(item_cats.columns.values)))

print('Shops : \n\t'+'\n\t'.join(shops.columns.tolist()))

print('Sales : \n\t'+'\n\t'.join(sales.columns.tolist()))

## you will get above data set along with row data of sales only in real world senario.

## based on those, Usually we have to create our training and test data set based on our model which we are going to use 

## Here they giving us and test data set where we can directly use and the sales data we can use for training the model

print('TestSet : \n\t'+'\n\t'.join(list(test)))

print('Output : \n\t'+'\n\t'.join(list(submission)))



sales.info()
print("Items")

print(items.head(2))

print("\nItem Catagerios")

print(item_cats.tail(2))

print("\nShops")

print(shops.sample(n=2))

print("\nTraining Data Set")

print(sales.sample(n=3,random_state=1))

print("\nTest Data Set")

print(test.sample(n=3,random_state=1))
from datetime import datetime

sales['year'] = sales['date'].dt.strftime('%Y')

sales['month'] = sales.date.apply(lambda x: x.strftime('%m')) #another way for same thing



sales.head(2)
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#will make your plot outputs appear and be stored within the notebook.

%matplotlib inline 



grouped = pd.DataFrame(sales.groupby(['year','month'])['item_cnt_day'].sum().reset_index())

sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grouped)

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts);
# Aggregate to monthly level the sales

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[

    "date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})



monthly_sales.head(5)
from numpy import array

# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)
sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()

#Keep only the test data of valid

sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')

#fill na with 0

sales_data_flat.fillna(0,inplace = True)

sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)

sales_data_flat.head(20)
#We will create pivot table.

# Rows = each shop+item code

# Columns will be out time sequence

pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )

pivoted_sales.head(20)
# X we will keep all columns execpt the last one 

X_train = np.expand_dims(pivoted_sales.values[:,:-1],axis = 2)

# the last column is our label

y_train = pivoted_sales.values[:,-1:]



# for test we keep all the columns execpt the first one

X_test = np.expand_dims(pivoted_sales.values[:,1:],axis = 2)



# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
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
submission_output = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_output = submission_output.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})

# creating csv file from dataframe

submission.to_csv('submission.csv',index = False)



submission.head()
X, y = split_sequence([10,20,30,40,50,60], 3)

#X, y = split_sequence(monthly_sales['item_cnt_day'], 3)

# summarize the data

for i in range(len(X)):

    print(X[i], y[i])