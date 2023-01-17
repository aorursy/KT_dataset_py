import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler



from subprocess import check_output

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))
import pypyodbc
# get data from mongodb

import io

import pymongo

from pymongo import MongoClient

import datetime



#client = MongoClient(connect=False) #Makes it "good enough" for our multi-threaded use case. 



# mng_client = pymongo.MongoClient('localhost', 27017)

# mng_db = mng_client['fx_prediction'] # Replace mongo db name

# collection_name = 'fx_tick_data_typed' # Replace mongo db table name

# db = mng_db[collection_name]



#print(db.count())

#min_date = datetime.datetime(2016, 1, 1, 0)

#max_date = datetime.datetime(2016, 12, 1, 0)

min_date = "1Jan16"

max_date = "1Feb16"



#https://bitbucket.org/djcbeach/monary/wiki/Home use to speed up
simname = "mine"
print(min_date)

print(max_date)
# # each of these is a stage in the pipeline - match, project, group, project.



# cursor_group = db.aggregate(

#    [

      

#        {"$match":{

#            "date": {

#                "$gte": min_date

#                , "$lte": max_date

#            }

#        }           

#        },

       

#        {

#            "$project": {

#                "_id" : 0

#                , "bo_spread": {"$subtract": ["$ask", "$bid"]}

#                , "bid": 1

#                , "ask": 1

#                , "date": 1

               

#            }

#        },

       

       

#        {

#         "$group" : {

#            #"_id" : "null",

#            "_id": {

#                "dateAgg": { "$dateToString": { "format": "%G/%m/%d %H:%M", "date": "$date" } }

#                },

#            #"high": { "$sum": { "$multiply": [ "$price", "$quantity" ] } },

#             "dateSample": {"$first": "$date"},

#             "high": { "$max": "$bid"},

#             "low": { "$min": "$bid"},

#             "open": { "$first": "$bid"},

#             "close": { "$last": "$bid"},

#            "avg_bo_spread": { "$avg": "$bo_spread" },

#            "max_bo_spread": { "$max": "$bo_spread" },

#            "min_bo_spread": { "$min": "$bo_spread" },

#            "count": { "$sum": 1 }

#         }

#       },

       

#        {

#            "$project": {

#                "_id" : 0

#                , "date": "$dateSample"

#                , "high": 1

#                , "low": 1

#                , "open": 1

#                , "close": 1

#                , "avg_bo_spread": 1

#                , "max_bo_spread": 1

#                , "min_bo_spread": 1

#                , "count": 1

#            }

#        }

       

#    ], allowDiskUse=True

# )

# cursor_group = db.aggregate(

#    [

#       {"$match":{

#            "date": {

#                "$gte": min_date

#                , "$lte": max_date

#            }

#        }           

#        },

       

             

#        {

#         "$group" : {

#            #"_id" : "null",

#            "_id": {

#                #"month": {"$month": "$date"}, 

#                #"day"  : {"$dayOfMonth": "$date"}, 

#                #"year" : {"$year": "$date"},

#                "time": { "$dateToString": { "format": "%G/%m/%d %H:%M", "date": "$date" } }

#                #"date": { "$dateFromParts": {"year": "$date", "month": "$date", "day": "$date", "hour": "$date", "minute": "$date"}}

#                },

#            #"high": { "$sum": { "$multiply": [ "$price", "$quantity" ] } },

#             "high": { "$max": "$date"},

#             "low": { "$min": "$date"},                                

#            "count": { "$sum": 1 }

#         }

#       }

#    ], allowDiskUse=True

# )





def getQueryRaw(strQuery, params=None, strConn=strConnDef, commitOn=None):



    if commitOn is None:

        commitOn = False



    if params is None:

        params = []



    pypyodbc.lowercase = False

    conn = pypyodbc.connect(strConn)

    cursor = conn.cursor()

    cursor.execute(strQuery, params)



    if commitOn:

        conn.commit()

        return "sql insert was successful.", "sql insert was successful."

    try:

        rows = cursor.fetchall()

        #print("rows", rows)

        # print("PARAMS:", params)

        description = cursor.description

        conn.close()

        return rows, description

    except:

        # print("THE QUERY: " + strQuery) TODO: add query

        conn.close()

        raise ValueError("There was an error fetching a sql query. Make sure the index exists for your selected dates. THE PARAMS: ", params)









def getQueryDataframe(strQuery, params=None, strConn=strConnDef, columnMustAlwaysExist=None, commitOn=None):



    rows, cursorDescription = getQueryRaw(strQuery, params, strConn, commitOn)

    if commitOn:

        return "sql insert was successful."



    if len(rows) == 0:

        print("No rows were returned.")

        print("THE PARAMS: ", params)

        print("THE QUERY: " + strQuery)

        print("Rows length is zero. No records returned")



        if columnMustAlwaysExist is None:

            columnMustAlwaysExist = "Empty"



        columns = ["Information", columnMustAlwaysExist]

        rows = [

            ["No results were returned.", "There is no data."]

            , ["No results were returned.", "There is no data."]

        ]



    else:

        # bytes conversion needed because of the linux pypyodbc bug

        columns = [column[0].decode("cp1252") if type(column[0]) == bytes else column[0] for column in

                   cursorDescription]



    results = pd.DataFrame(data=rows, columns=columns)





    return results

str_query = """





select

    --distinct

    const.year, const.month, const.day, const.hour, const.weekday, round(const.minute/15,0) * 15

    , const.snaptime 'date'

    , const.bid_price

    , const.ask_price

    , const.ask_price - const.bid_price 'bo_spread'

	, max(const.bid_price) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0))'high'

	, min(const.bid_price) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0)) 'low'

    , avg(const.ask_price - const.bid_price) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0)) 'avg_bo_spread'

	--, min(const.snaptime) 'open_datetime'

	--, max(const.snaptime) 'close_datetime'

	, count(*) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0)) 'count'

    , first_value(const.bid_price) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0) order by const.snaptime) 'open'

    , last_value(const.bid_price) over (partition by const.year, const.month, const.day, const.hour, round(const.minute/15,0) order by const.snaptime) 'close'

from dbo.fx_spot_data_features const

where

    const.snaptime >= '"""+min_date+"""'

    and const.snaptime <= '"""+max_date+"""'

    

--group by const.year, const.month, const.day, const.hour, round(const.minute/15,0)

--order by const.year, const.month, const.day, const.hour, round(const.minute/15,0)

order by const.snaptime



"""

res = getQueryDataframe(str_query)

print(res.count())

res.head()
#cursor = list(cursor_group)

df_res =  pd.DataFrame(res)

df_res.set_index('date', inplace=True)
df_res.to_csv("eurusd_features.csv")
df_res = pd.read_csv("data/eurusd_features.csv")

df_res.set_index('date', inplace=True)
# load kaggle reference dataset for comparison

#df_kaggle = pd.read_csv('data/bm_kaggle/EURUSD_15m_BID_sample.csv')

df_kaggle = pd.read_csv('data/bm_kaggle/EURUSD_15m_BID_01.01.2010-31.12.2016.csv')



# Rename bid OHLC columns

df_kaggle.rename(columns={'Time' : 'date', 'Open' : 'open', 'Close' : 'close', 

                   'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Volume' : 'volume'}, inplace=True)

df_kaggle['date'] = pd.to_datetime(df_kaggle['date'], infer_datetime_format=True)

df_kaggle.set_index('date', inplace=True)

df_kaggle = df_kaggle.astype(float)



simname = "bm_kaggle"



df_res = df_kaggle
#df_res.date = pd.to_datetime(df_res.date.dateAgg, format='%Y%m%d %H:%M')

df_res.head()
df = df_res

# to include seasonality as a feature

#df['hour'] = df.index.hour

#df['day']  = df.index.weekday

#df['week'] = df.index.week

#df['month'] = df.index.month



#df['momentum']  = df['volume'] * (df['open'] - df['close'])

df['avg_price'] = (df['low'] + df['high'])/2

df['range']     = df['high'] - df['low']

df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close'])/4

df['oc_diff']    = df['open'] - df['close']

#df['bo_spread'] = df.ask - df.bid

df['period_return'] = df.close / df.open
# create ohlc prices, analyse distribution, think about feature transformation and de-trending



fig, axarr = plt.subplots(2, 5, figsize=(30,10)) #1 row, 2 cols, x, y

#plt.figure(figsize=(20, 4))

i_row, i_col = 0,0

fig.suptitle("frequency distributions")





sns.distplot(df.period_return-1, ax=axarr[i_row, i_col])

#axarr[0, 0].set_title('Axis [0,0] Subtitle')



i_col += 1

sns.distplot(df.bo_spread, ax=axarr[i_row, i_col])



i_col += 1

sns.distplot(df.avg_bo_spread, ax=axarr[i_row, i_col])



x_axis_col = "ohlc_price"

y_axis_col = "bo_spread"

i_col += 1

norm = colors.Normalize(df[x_axis_col].values.min(), df[x_axis_col].values.max())

color = cm.viridis(norm(df[x_axis_col].values))

axarr[i_row, i_col].scatter(df[x_axis_col].values, df[y_axis_col].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

axarr[i_row, i_col].set_xlabel(x_axis_col)







x_axis_col = "period_return"

y_axis_col = "bo_spread"

i_col += 1

norm = colors.Normalize(df[x_axis_col].values.min(), df[x_axis_col].values.max())

color = cm.viridis(norm(df[x_axis_col].values))

axarr[i_row, i_col].scatter(df[x_axis_col].values, df[y_axis_col].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

axarr[i_row, i_col].set_xlabel(x_axis_col)





i_row, i_col = 1, 0 # move down one row



x_axis_col = "hour"

y_axis_col = "bo_spread"

norm = colors.Normalize(df[x_axis_col].values.min(), df[x_axis_col].values.max())

color = cm.viridis(norm(df[x_axis_col].values))

axarr[i_row, i_col].scatter(df[x_axis_col].values, df[y_axis_col].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

axarr[i_row, i_col].set_xlabel(x_axis_col)





x_axis_col = "hour"

y_axis_col = "count"

i_col += 1

norm = colors.Normalize(df[x_axis_col].values.min(), df[x_axis_col].values.max())

color = cm.viridis(norm(df[x_axis_col].values))

axarr[i_row, i_col].scatter(df[x_axis_col].values, df[y_axis_col].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

axarr[i_row, i_col].set_xlabel(x_axis_col)



x_axis_col = "day"

y_axis_col = "count"

i_col += 1

norm = colors.Normalize(df[x_axis_col].values.min(), df[x_axis_col].values.max())

color = cm.viridis(norm(df[x_axis_col].values))

axarr[i_row, i_col].scatter(df[x_axis_col].values, df[y_axis_col].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

axarr[i_row, i_col].set_xlabel(x_axis_col)





#plt.tight_layout() # reduce overlap

plt.show()



print("Nb Rows: ", df.high.count())
# all at once

sns.pairplot(df, hue="bo_spread")
import dill as pickle

with open(simname+'_eurusd_features.pkl', 'wb') as file:

    pickle.dump(df, file)
# Add PCA as a feature instead of for reducing the dimensionality. This improves the accuracy a bit.

from sklearn.decomposition import PCA



dataset = df.copy().values.astype('float32')

pca_features = df.columns.tolist()



pca = PCA(n_components=1)

df['pca'] = pca.fit_transform(dataset)
import matplotlib.colors as colors

import matplotlib.cm as cm

import pylab



plt.figure(figsize=(10,5))

norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())

color = cm.viridis(norm(df['ohlc_price'].values))

plt.scatter(df['ohlc_price'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

plt.title('ohlc_price vs pca')

plt.show()



if simname != "bm_kaggle":



    plt.figure(figsize=(10,5))

    norm = colors.Normalize(df['avg_bo_spread'].values.min(), df['avg_bo_spread'].values.max())

    color = cm.viridis(norm(df['avg_bo_spread'].values))

    plt.scatter(df['avg_bo_spread'].values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

    plt.title('avg_bo_spread vs pca')

    plt.show()





    plt.figure(figsize=(10,5))

    norm = colors.Normalize(df['avg_bo_spread'].values.min(), df['avg_bo_spread'].values.max())

    color = cm.viridis(norm(df['avg_bo_spread'].values))

    plt.scatter(df['avg_bo_spread'].values, df['ohlc_price'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

    plt.title('avg_bo_spread vs ohlc_price')

    plt.show()





    plt.figure(figsize=(10,5))

    norm = colors.Normalize(df['avg_bo_spread'].values.min(), df['avg_bo_spread'].values.max())

    color = cm.viridis(norm(df['avg_bo_spread'].values))

    plt.scatter(df['avg_bo_spread'].values, df['period_return'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

    plt.title('avg_bo_spread vs period_return')

    plt.show()

    

    plt.figure(figsize=(10,5))

    norm = colors.Normalize(df['avg_bo_spread'].values.min(), df['avg_bo_spread'].values.max())

    color = cm.viridis(norm(df['avg_bo_spread'].values))

    plt.scatter(df['avg_bo_spread'].values, df['period_return'].shift().values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

    plt.title('avg_bo_spread vs period_return shift')

    plt.show()

    

    plt.figure(figsize=(10,5))

    norm = colors.Normalize(df['bo_spread'].values.min(), df['bo_spread'].values.max())

    color = cm.viridis(norm(df['bo_spread'].values))

    plt.scatter(df['bo_spread'].values, df['period_return'].shift().values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

    plt.title('bo_spread vs period_return shift')

    plt.show()







plt.figure(figsize=(10,5))

norm = colors.Normalize(df['ohlc_price'].values.min(), df['ohlc_price'].values.max())

color = cm.viridis(norm(df['ohlc_price'].values))

plt.scatter(df['ohlc_price'].shift().values, df['pca'].values, lw=0, c=color, cmap=pylab.cm.cool, alpha=0.3, s=1)

plt.title('ohlc_price - 15min future vs pca')

plt.show()

# this creates a training dataset for the model

def create_dataset(dataset, look_back=20):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back)]

        dataX.append(a)

        dataY.append(dataset[i + look_back])

    return np.array(dataX), np.array(dataY)
# check feature correlation, to see what correlates with the close price

colormap = plt.cm.inferno

plt.figure(figsize=(15,15))

plt.title('Pearson correlation of features', y=1.05, size=15)

sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()



plt.figure(figsize=(15,5))

corr = df.corr()

sns.heatmap(corr[corr.index == 'close'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
# create random forest regressor - random decision trees, like weak learner, ada boost

from sklearn.ensemble import RandomForestRegressor



# Scale and create datasets

target_index = df.columns.tolist().index('close')

dataset = df.values.astype('float32')



# Scale the data

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



# Set look_back to 20 which is 5 hours (15min*20)

X, y = create_dataset(dataset, look_back=1)

y = y[:,target_index]

X = np.reshape(X, (X.shape[0], X.shape[2]))
# fit model

forest = RandomForestRegressor(n_estimators = 5)

forest = forest.fit(X, y)
# find feature with best explanatory power to predict close price

importances = forest.feature_importances_

std = np.std([forest.feature_importances_ for forest in forest.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



column_list = df.columns.tolist()

print("Feature ranking:")

for f in range(X.shape[1]-1):

    print("%d. %s %d (%f)" % (f, column_list[indices[f]], indices[f], importances[indices[f]]))



# Plot the feature importances coming from the forest of decision trees

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="salmon", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
# plot close price, compare to low and high price

ax = df.plot(x=df.index, y='close', c='red', figsize=(40,10))

index = [str(item) for item in df.index]

plt.fill_between(x=index, y1='low',y2='high', data=df, alpha=0.4)

plt.show()



# plot first 200 entries 

p = df[:200].copy()

ax = p.plot(x=p.index, y='close', c='red', figsize=(40,10))

index = [str(item) for item in p.index]

plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)

plt.title('zoomed, first 200')

plt.show()
# Scale and create datasets

target_index = df.columns.tolist().index('close')

high_index = df.columns.tolist().index('high')

low_index = df.columns.tolist().index('low')

dataset = df.values.astype('float32')



# Scale the data

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



# Create y_scaler to inverse it later

y_scaler = MinMaxScaler(feature_range=(0, 1))

t_y = df['close'].values.astype('float32')

t_y = np.reshape(t_y, (-1, 1))

y_scaler = y_scaler.fit(t_y)

    

# Set look_back to 20 which is 5 hours (15min*20)

X, y = create_dataset(dataset, look_back=1)

y = y[:,target_index]
# Set training data size

# We have a large enough dataset. So divid into 98% training / 1%  development / 1% test sets

train_size = int(len(X) * 0.99)

trainX = X[:train_size]

trainY = y[:train_size]

testX = X[train_size:]

testY = y[train_size:]
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense



# create a small LSTM network

model = Sequential()

model.add(LSTM(20, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

model.add(LSTM(20, return_sequences=True))

model.add(LSTM(10, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(4, return_sequences=False))

model.add(Dense(4, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='relu'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])

print(model.summary())


# Save the best weight during training.

simname = "bm_kaggle_5"

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(simname + ".weights.best.hdf5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')



# Fit

callbacks_list = [checkpoint]

%time history = model.fit(trainX, trainY, epochs=100, batch_size=10000, verbose=0, callbacks=callbacks_list, validation_split=0.1)
epoch = len(history.history['loss'])

print("epoch", epoch)

for k in list(history.history.keys()):

    if 'val' not in k:

        plt.figure(figsize=(40,10))

        plt.plot(history.history[k])

        plt.plot(history.history['val_' + k])

        plt.title(k)

        plt.ylabel(k)

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()
min(history.history['val_mean_absolute_error'])
# tune model by starting from best weights and rerunning with decaying learning rate

# Load the weight that worked the best

model.load_weights(simname+".weights.best.hdf5")

#epoch=60



# Train again with decaying learning rate

from keras.callbacks import LearningRateScheduler

import keras.backend as K



def scheduler(epoch):

    if epoch%2==0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr*.9)

        print("lr changed to {}".format(lr*.9))

    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)



callbacks_list = [checkpoint, lr_decay]

history = model.fit(trainX, trainY, epochs=int(epoch/3), batch_size=10000, verbose=0, callbacks=callbacks_list, validation_split=0.1)
epoch = len(history.history['loss'])

for k in list(history.history.keys()):

    if 'val' not in k:

        plt.figure(figsize=(40,10))

        plt.plot(history.history[k])

        plt.plot(history.history['val_' + k])

        plt.title(k)

        plt.ylabel(k)

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

min(history.history['val_mean_absolute_error'])
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Benchmark

model.load_weights(simname+".weights.best.hdf5")



pred = model.predict(testX)



predictions = pd.DataFrame()

predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))

predictions['actual'] = testY

predictions = predictions.astype(float)



predictions.plot(figsize=(20,10))

plt.show()



predictions['diff'] = predictions['predicted'] - predictions['actual']

plt.figure(figsize=(10,10))

sns.distplot(predictions['diff']);

plt.title('Distribution of differences between actual and prediction')

plt.show()



print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['actual'].values))

print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))

predictions['diff'].describe()
pred = model.predict(testX)

pred = y_scaler.inverse_transform(pred)

close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))

predictions = pd.DataFrame()

predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))

predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))



p = df[-pred.shape[0]:].copy()

predictions.index = p.index

predictions = predictions.astype(float)

predictions = predictions.merge(p[['low', 'high']], right_index=True, left_index=True)



ax = predictions.plot(x=predictions.index, y='close', c='red', figsize=(40,10))

ax = predictions.plot(x=predictions.index, y='predicted', c='blue', figsize=(40,10), ax=ax)

index = [str(item) for item in predictions.index]

plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)

plt.title('Prediction vs Actual (low and high as blue region)')

plt.show()



predictions['diff'] = predictions['predicted'] - predictions['close']

plt.figure(figsize=(10,10))

sns.distplot(predictions['diff']);

plt.title('Distribution of differences between actual and prediction ')

plt.show()



g = sns.jointplot("diff", "predicted", data=predictions, kind="kde", space=0)

plt.title('Distributtion of error and price')

plt.show()



# predictions['correct'] = (predictions['predicted'] <= predictions['high']) & (predictions['predicted'] >= predictions['low'])

# sns.factorplot(data=predictions, x='correct', kind='count')



print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['close'].values))

print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['close'].values))

predictions['diff'].describe()
