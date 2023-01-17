import datetime
import os
import pandas as pd
import math
from statsmodels.tsa.arima_model import ARMA
# full timeframe covered in the dataset
FULL_MONTH_SET = pd.date_range(start='1/1/2013', periods=34,freq=pd.offsets.MonthBegin(1))
# month wee are trying to predict
TARGET = datetime.date(2015, 11, 1)

traindf = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
traindf.index = pd.to_datetime(traindf['date'], format='%d.%m.%Y')
# extracts data about a single item
def getCountByItemId(itemid,data):
    item_data = data.loc[traindf['item_id'] == itemid]
    result = {}
    for store in item_data.shop_id.unique():
        store_fitered = item_data.loc[item_data['shop_id'] == store]
        store_fitered = store_fitered.groupby(by=[store_fitered.index.month, store_fitered.index.year]).sum()
        store_fitered = store_fitered.drop(['date_block_num', 'shop_id', 'item_id', 'item_price'], axis=1)
        store_fitered.index = pd.to_datetime(store_fitered.index, format='(%m, %Y)')
        result[store] = store_fitered
    return result

# function for completing the data for missing month of sales with zeros
def complete_data(data):
    for key in data.keys():
        for date in FULL_MONTH_SET:
            if date not in data[key].index:
                data[key].loc[date] = 0
        data[key] = data[key].sort_index()
    return data

# function for getting a single item prediction model
def getSingleSimplePredictor(itemdata):
    model = ARMA(itemdata,order=(0,1))
    model = model.fit(disp=False)
    return model

# function for getting dictionary of models for each item indexed item_id=>model
def getModels(dataset):
    result = {}
    count = 0
    for item in dataset.item_id.unique():
        print(str(count)+'/'+str(len(dataset.item_id.unique())))
        count+=1
        item_data = getCountByItemId(item, dataset)
        item_data = complete_data(item_data)
        for key in item_data.keys():
            if item_data[key].item_cnt_day.mean() != 0:
                item_data[key] = getSingleSimplePredictor(item_data[key])
        result[item] = item_data
    return result

def getPrediction(model,shop_id,item_id):
    res = 0
    if item_id in model.keys():
        if shop_id in model[item_id].keys() :
            try:
                res = model[item_id][shop_id].predict(TARGET)[0]
            except:
                return res
    return res
# building the models
models = getModels(traindf)
# building the prediction to format in the competition 
test_task = pd.read_csv('test.csv')
prediction = pd.DataFrame(columns=['ID','item_cnt_month'])
stores = test_task.shop_id.values
items = test_task.item_id.values
maxID = len(test_task)
for i in range(maxID):
    prediction = prediction.append({'ID': i, 'item_cnt_month': getPrediction(models,stores[i],items[i])},
                                   ignore_index=True)
prediction.to_csv('res.csv',index=False)

FULL_MONTH_SET = pd.date_range(start='1/1/2013', periods=34,freq=pd.offsets.MonthBegin(1))

# function for completing the data for missing month of sales with zeros
def complete_data(data):
    shop = data.shop_id[0]
    item = data.item_id[0]
    for date in FULL_MONTH_SET:
        if date not in data.index:
            data.loc[date] = 0
    data['shop_id'] = shop
    data['item_id'] = item
    data['date'] = [str(date.month)+"-"+str(date.year) for date in data.index]
    return data


traindf = pd.read_csv('sales_train.csv')
traindf.index = pd.to_datetime(traindf['date'], format='%d.%m.%Y')
groupeddata = []
# building sub frames with grouped items by month and item, concating them in the end
for itemid in traindf.item_id.unique():
    item_data = traindf.loc[traindf['item_id'] == itemid]
    for store in item_data.shop_id.unique():
        store_fitered = item_data.loc[item_data['shop_id'] == store]
        store_fitered = store_fitered.groupby(by=[store_fitered.index.month, store_fitered.index.year]).sum()
        store_fitered = store_fitered.drop(['date_block_num','item_price'], axis=1)
        store_fitered['item_id']=itemid
        store_fitered['shop_id']=store
        store_fitered.index = pd.to_datetime(store_fitered.index, format='(%m, %Y)')
        groupeddata.append(complete_data(store_fitered))
result = pd.concat(groupeddata)
result.to_csv('../input/preprocessed/grouped_data.csv',index=False)

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Input, Model
from keras.layers import Embedding, Flatten, Concatenate, Dropout, Dense
from keras.optimizers import Adamax


result = pd.read_csv('../input/preprocessed/grouped_data.csv')
result = result.rename(columns={"date": "month", "date.1": "year"})
result.index = pd.to_datetime(result['month'], format='%m-%Y')
print(result.info(verbose=True))

converterValue = result['item_cnt_day'].max()

# Normalizing inputs
Shops = result['shop_id']
Shops = Shops.values
Items = result['item_id']
Items = Items.values
Month = result.index.month - 1
Month = Month.values
Year = result.index.year - 2013
Year = Year.values
Sales = result['item_cnt_day'] * (1 / converterValue)
Sales = Sales.values
shop_input = Input(name='shop', shape=[1])
item_input = Input(name='item', shape=[1])
month_input = Input(name='month', shape=[1])
year_input = Input(name='year', shape=[1])


shops_embeding = Embedding(input_dim=result['shop_id'].unique().max(), output_dim=25,  input_length=1)(shop_input)
item_embeding = Embedding(input_dim=result['item_id'].unique().max(), output_dim=25,  input_length=1)(item_input)
month_embeding = Embedding(input_dim=11, output_dim=5,  input_length=1)(month_input)
year_embeding = Embedding(input_dim=3, output_dim=2,  input_length=1)(year_input)

user_flaten = Flatten()(shops_embeding)
item_flaten = Flatten()(item_embeding)
month_flaten = Flatten()(month_embeding)
year_flaten = Flatten()(year_embeding)


conc = Concatenate()([user_flaten, item_flaten, month_flaten, year_flaten])
hid1 = Dense(200, activation='relu')(conc)
drop = Dropout(0.2)(hid1)
hid2 = Dense(50, activation='relu')(conc)
prediction = Dense(1, activation='relu')(hid2)


model = Model(inputs=[shop_input, item_input, month_input, year_input], outputs=prediction)
model.summary()

model.compile(loss='mse', optimizer=Adamax(lr=0.001), metrics=['mae'])
# using callbacks
callbacks = [EarlyStopping('val_loss', patience=10), ModelCheckpoint("model_backup.h5", save_best_only=True)]
history = model.fit([Shops, Items, Month,Year], Sales, epochs=50, validation_split=.2, verbose=1, callbacks=callbacks,
                    batch_size = 500)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
test['month'] = 10
test['year'] = 2

test_Shops = test['shop_id']
test_Shops = test_Shops.values
test_Items = test['item_id']
test_Items = test_Items.values
test_Month = test['month']
test_Month = test_Month.values
test_Year = test['year']
test_Year = test_Year.values

prediction = model.predict([test_Shops, test_Items, test_Month, test_Year])
divider = 1 / converterValue
result = pd.DataFrame()
result['item_cnt_month'] = [math.ceil(x[0]/divider) for x in prediction]
result['ID'] = result.index
result.to_csv('res7.csv',index=False)


from datetime import datetime

traindata = pd.read_csv('../input/preprocessed/grouped_data.csv')
item_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
# encoding the month into seasons
def getSeason(month):
    if month < 2 or month == 11:
        return 0
    elif month < 5:
        return 1
    elif month < 8:
        return 2
    else:
        return 3

# features: category, year, month of the year, quarter, month of the quarter, season
# category
joined_data = pd.merge(left=traindata,right=item_data,on='item_id')
joined_data = joined_data.drop(columns=['item_name'])

joined_data.index = pd.to_datetime(joined_data['date'], format='%m-%Y')
# year
joined_data['year'] = joined_data.index.year

# month of the year
joined_data['month'] = joined_data.index.month

# quarter
joined_data['quarter'] = [math.floor(month/4+1) for month in joined_data.index.month]

# month of the quarter
joined_data['m_quarter'] = joined_data.index.month%4+1

# season
joined_data['season'] = [getSeason(month) for month in joined_data.index.month]

joined_data.to_csv('enriched_grouped_data.csv',index=False)

enriched_grouped_data = pd.read_csv('enriched_grouped_data.csv')

# Building extended inputs
Shops = enriched_grouped_data['shop_id']
Shops = Shops.values

Items = enriched_grouped_data['item_id']
Items = Items.values

Month = enriched_grouped_data['month'] - 1
Month = Month.values

Year = enriched_grouped_data['year'] - 2013
Year = Year.values

Category = enriched_grouped_data['item_category_id']
Category = Category.values

Sales = enriched_grouped_data['item_cnt_day'] * (1 / converterValue)
Sales = Sales.values

input_data = [Shops, Items, Category, Month, Year]
shop_input = Input(name='shop', shape=[1])
item_input = Input(name='item', shape=[1])
month_input = Input(name='month', shape=[1])
year_input = Input(name='year', shape=[1])
category_input = Input(name='category', shape=[1])

shops_embeding = Embedding(input_dim=enriched_grouped_data['shop_id'].unique().max(), output_dim=15,  input_length=1)(shop_input)
item_embeding = Embedding(input_dim=enriched_grouped_data['item_id'].unique().max(), output_dim=15,  input_length=1)(item_input)
cat_embeding = Embedding(input_dim=enriched_grouped_data['item_category_id'].unique().max(), output_dim=10,  input_length=1)(category_input)
month_embeding = Embedding(input_dim=11, output_dim=12,  input_length=1)(month_input)
year_embeding = Embedding(input_dim=3, output_dim=2,  input_length=1)(year_input)

shop_flaten = Flatten()(shops_embeding)
item_flaten = Flatten()(item_embeding)
month_flaten = Flatten()(month_embeding)
year_flaten = Flatten()(year_embeding)
cat_flaten = Flatten()(cat_embeding)


conc = Concatenate()([shop_flaten, item_flaten, cat_flaten, month_flaten, year_flaten])
hid1 = Dense(200, activation='relu')(conc)
drop = Dropout(0.2)(hid1)
hid2 = Dense(5, activation='relu')(conc)
prediction = Dense(1, activation='relu')(hid2)


upgraded_model = Model(inputs=[shop_input, item_input,category_input, month_input, year_input], outputs=prediction)
upgraded_model.summary()
upgraded_model.compile(loss='mse', optimizer=Adamax(lr=0.001), metrics=['mae'])
# using callbacks
callbacks = [EarlyStopping('val_loss', patience=5), ModelCheckpoint("model_backup.h5", save_best_only=True)]
history = upgraded_model.fit(input_data, Sales, epochs=50, validation_split=.2, verbose=1, callbacks=callbacks,
                    batch_size = 500)
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
item_data = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
joined_test = pd.merge(left=test,right=item_data,on='item_id')

joined_test['year'] = 2
joined_test['month'] = 10
joined_test['season'] = 3


test_Shops = joined_test['shop_id']
test_Shops = test_Shops.values
test_Items = joined_test['item_id']
test_Items = test_Items.values
test_Categories = joined_test['item_category_id']
test_Categories = test_Categories.values
test_Month = joined_test['month']
test_Month = test_Month.values
test_Year = joined_test['year']
test_Year = test_Year.values
test_Season = joined_test['season']
test_Season = test_Season.values

prediction = upgraded_model.predict([test_Shops, test_Items, test_Categories, test_Month, test_Year])

divider = 1 / converterValue
result = pd.DataFrame()
result['item_cnt_month'] = [math.ceil(x[0]) for x in prediction]
result['ID'] = result.index
result.to_csv('res6.csv',index=False)
cut_model = Model(inputs=[shop_input, item_input,category_input, month_input, year_input],outputs=hid2)
cut_model.summary()
parameter_prediction = cut_model.predict([Shops, Items, Category, Month, Year])
print(parameter_prediction)
classical_ml_dataset = pd.DataFrame()
for i in range(len(parameter_prediction[0])):
    classical_ml_dataset[str(i)] = [x[i] for x in parameter_prediction]
classical_ml_dataset['target'] = enriched_grouped_data['item_cnt_day']
y = classical_ml_dataset['target']
X = classical_ml_dataset.iloc[:,0:5]
X.head()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


# we will use SVR
model = LinearRegression( )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)
model.fit(X_train,y_train)
score = model.score(X_test, y_test)
print(score)
test_prediction = cut_model.predict([test_Shops, test_Items, test_Categories, test_Month, test_Year])
classical_ml_result = pd.DataFrame()
for i in range(len(test_prediction[0])):
    classical_ml_result[str(i)] = [x[i] for x in test_prediction]
classical_ml_result.head()
classical_ml_prediction = model.predict(classical_ml_result)
result = pd.DataFrame()
result['item_cnt_month'] = [x for x in classical_ml_prediction]
result['ID'] = result.index
result.to_csv('res8.csv',index=False)