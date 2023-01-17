import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
print(train.head())

cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
print(cats.head())
df_cat = train[['date_block_num', 'item_id', 'item_price']]
print(df_cat.head())
print(df_cat.columns)
print(cats.columns)
df_cat= pd.merge(df_cat, cats, on='item_id')
df_cat['cat_count']=1
print(df_cat)
df_cat=df_cat[['item_category_id','item_price']]
df_cat=df_cat.groupby('item_category_id').sum()
df_cat=df_cat.sort_values('item_price',ascending=False)
important_category=df_cat.head(5)
print(important_category)
df_cat=[]
print(important_category.index[:])
df_cat = train[['date_block_num', 'item_id', 'item_price']]
df_cat= pd.merge(df_cat, cats, on='item_id')
df_cat=df_cat.groupby(['date_block_num', 'item_category_id']).size()

df_item_price = train.groupby('date_block_num')['item_price'].sum()

rows=32
cols=83

input_matrix=np.empty(shape=(rows,len(important_category.index)), dtype=int)

for x in range(32):
    for y in range(len(important_category.index)):
        if pd.isnull(df_cat.get(x).get(important_category.index[y])):
            input_matrix[x][y]=0
        else:
            input_matrix[x][y]=df_cat.get(x).get(important_category.index[y])
            
print(input_matrix)
label_matrix = np.empty(shape=(rows), dtype=int)
for count in range(32):
    if df_item_price[count+1]-df_item_price[count] > 0:
        label_matrix[count]=1
    else:
        label_matrix[count]=0
        
print(label_matrix)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(input_matrix, label_matrix, epochs=700, batch_size=1)
_, accuracy = model.evaluate(input_matrix, label_matrix)
print('Accuracy: %.2f' % (accuracy*100))
