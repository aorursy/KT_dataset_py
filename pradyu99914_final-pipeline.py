# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.|

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install holidays



#package imports

import pandas as pd

from sklearn import preprocessing

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from bayes_opt import BayesianOptimization

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

import seaborn as sns

from sklearn.model_selection import train_test_split





import gc

import os

from tqdm import tqdm

import holidays

import datetime as dt

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline
us_holidays = holidays.US()

def haversine_distance(lat1, long1, lat2, long2):

    R = 6371  #radius of earth in kilometers

    phi1 = np.radians(lat1)

    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2-lat1)

    delta_lambda = np.radians(long2-long1)

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c

    d = (R * c) #in kilometers

    return d



def readData(nrows = 15000000):

    

    #kaggle kernels: https://www.kaggle.com/pradyu99914/data-feature-engineering?scriptVersionId=21782913

    #feature engineering: https://www.kaggle.com/anushkini/nyc-taxi-fare-graphs

    

    try:

        test_df = pd.read_feather("/kaggle/input/final-pipeline/test_df.feather")

        df_chunk = pd.read_feather("/kaggle/input/final-pipeline/df_chunk.feather")

    except Exception:

        #read the test and train sets

        gc.collect()

        df_chunk = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 15_000_000)

        test_df = pd.read_feather('../input/data-feature-engineering/test_feature.feather')

        gc.collect()





        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)

        df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

        df_chunk.dropna()

        #remove the rows that have coordinates outside the bounding box of the city and its nearby areas.

        mask = df_chunk['pickup_longitude'].between(-75, -73)

        mask &= df_chunk['dropoff_longitude'].between(-75, -73)

        mask &= df_chunk['pickup_latitude'].between(40, 42)

        mask &= df_chunk['dropoff_latitude'].between(40, 42)

        #remove the rows that have wrong number of passengers(negative or more than 8 passsengers)

        mask &= df_chunk['passenger_count'].between(0, 8)

        #remove rows with wrong fares(negative fares and grater than 250 USD..) and rows with fare amount = 0

        mask &= df_chunk['fare_amount'].between(0, 250)

        mask &= df_chunk['fare_amount'].gt(0)



        #apply this mask, which will remove all the inconsistent rows

        df_chunk = df_chunk[mask]

        #print("After: ",len(df_chunk))

        df_chunk = df_chunk.reset_index()  #make it featherable again. masking messes with the index. reset index helps remove this problem.

        mask = 0

        #recover memory!

        gc.collect()

        

        #add time and holiday features

        df_chunk["time"] = pd.to_numeric(df_chunk.apply(lambda r: r.pickup_datetime.hour*60 + r.pickup_datetime.minute, axis = 1), downcast = "unsigned")

        gc.collect()

        df_chunk["holiday"] = pd.to_numeric(df_chunk.apply(lambda x: 1 if x.pickup_datetime.strftime('%d-%m-%y')in us_holidays else 0, axis =1), downcast = "unsigned")

        gc.collect()

        

        #coordinates for important places in the city

        Manhattan = (-73.9712,40.7831)[::-1]

        JFK_airport = (-73.7781,40.6413)[::-1]

        Laguardia_airport = (-73.8740,40.7769)[::-1]

        statue_of_liberty = (-74.0445,40.6892)[::-1]

        central_park = (-73.9654,40.7829)[::-1]

        time_square = (-73.9855,40.7580)[::-1]

        brooklyn_bridge = (-73.9969,40.7061)[::-1]

        rockerfeller = (-73.9787,40.7587)[::-1]



        #more features

        df_chunk["distance"] = pd.to_numeric(haversine_distance(df_chunk['pickup_latitude'], df_chunk['pickup_longitude'], df_chunk['dropoff_latitude'], df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk["year"] = df_chunk["pickup_datetime"].dt.year

        df_chunk["weekday"] = pd.to_numeric(df_chunk["pickup_datetime"].dt.weekday, downcast= "unsigned")



        #distance from tourist spots

        df_chunk['pickup_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_Mtn'] = pd.to_numeric(haversine_distance(Manhattan[0],Manhattan[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['pickup_distance_jfk'] = pd.to_numeric(haversine_distance(JFK_airport[0],JFK_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['pickup_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_lg'] = pd.to_numeric(haversine_distance(Laguardia_airport[0],Laguardia_airport[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')



        #add the date and month features.

        df_chunk['day'] = df_chunk['pickup_datetime'].dt.day

        df_chunk['month'] = df_chunk['pickup_datetime'].dt.month



        test_df['day'] = test_df['pickup_datetime'].dt.day

        test_df['month'] = test_df['pickup_datetime'].dt.month



        #add more distances from tourist spots

        df_chunk['pickup_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['pickup_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['pickup_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['pickup_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')

        df_chunk['pickup_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],df_chunk['pickup_latitude'],df_chunk['pickup_longitude']), downcast = 'float')

        df_chunk['dropoff_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],df_chunk['dropoff_latitude'],df_chunk['dropoff_longitude']), downcast = 'float')



        test_df['pickup_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

        test_df['dropoff_distance_sol'] = pd.to_numeric(haversine_distance(statue_of_liberty[0],statue_of_liberty[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

        test_df['pickup_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

        test_df['dropoff_distance_cp'] = pd.to_numeric(haversine_distance(central_park[0],central_park[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

        test_df['pickup_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

        test_df['dropoff_distance_ts'] = pd.to_numeric(haversine_distance(time_square[0],time_square[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

        test_df['pickup_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

        test_df['dropoff_distance_bb'] = pd.to_numeric(haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')

        test_df['pickup_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],test_df['pickup_latitude'],test_df['pickup_longitude']), downcast = 'float')

        test_df['dropoff_distance_r'] = pd.to_numeric(haversine_distance(rockerfeller[0],rockerfeller[1],test_df['dropoff_latitude'],test_df['dropoff_longitude']), downcast = 'float')



        df_chunk['pickup_longitude'] = np.radians(df_chunk['pickup_longitude'])

        df_chunk['pickup_latitude'] = np.radians(df_chunk['pickup_latitude'])

        df_chunk['dropoff_latitude'] = np.radians(df_chunk['dropoff_latitude'])

        df_chunk['dropoff_longitude'] = np.radians(df_chunk['dropoff_longitude'])



        test_df['pickup_longitude'] = np.radians(test_df['pickup_longitude'])

        test_df['pickup_latitude'] = np.radians(test_df['pickup_latitude'])

        test_df['dropoff_latitude'] = np.radians(test_df['dropoff_latitude'])

        test_df['dropoff_longitude'] = np.radians(test_df['dropoff_longitude'])

        

        

    #write this back so that it will be availablwe after the next commit

    test_df.to_feather("test_df.feather")

    df_chunk.to_feather("df_chunk.feather")

    y = df_chunk['fare_amount']

    

    #drop the unwanted columns

    df_chunk = df_chunk.drop(['key','pickup_datetime','fare_amount'],axis = 1)

    X_train,X_val,y_train,y_val = train_test_split(df_chunk,y,test_size = 0.1)

    del X_train['index']

    test_df = test_df[X_train.columns]

    del(df_chunk)

    del(y)

    gc.collect()



    if nrows!=15000000: #reading lesser number of rows

        X_train = X_train[:nrows]

        y_train = y_train[:nrows]

        gc.collect()



    return X_train, y_train, X_val, y_val, test_df
# transform the data into lgbm-compatible form

X_train, y_train, X_val, y_val, test_df = readData(1500000) #number of data points to read, please make sure this is atleast 50k as some models reserve some data for grid search
test_df.head()
#check if the model is already present, if not train it again.

def getLGB():

    #kaggle kernel - https://www.kaggle.com/anushkini/taxi-lightgbm?scriptVersionId=23609067

    try:

        #try to read the model

        model = lgb.Booster(model_file = "/kaggle/input/trained-model/model.txt" )

    except Exception:

        #if the trained model is not present, train it again

        lgbm_params =  {

            'task': 'train',

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'nthread': 4,

            'learning_rate': 0.05,

            'bagging_fraction': 1,

            'num_rounds':50000

            }

        model = lgb.train(lgbm_params, train_set = dtrain, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=dval)

        del(X_train)

        del(y_train)

        del(X_val)

        del(y_val)

        gc.collect()

    

    return model
def getkNNpredictions(X_train, y_train, X_test):

    #kaggle kernel - https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models-knn?scriptVersionId=22570753

    from sklearn.neighbors import KNeighborsRegressor

    #this will store the predictions for each of the knn regressors

    knnregressoroutputs = []

    #go through chunks of 1M

    for i in tqdm(range(len(X_train)//1000000)):

        neigh = KNeighborsRegressor(n_neighbors=2)

        #extract the required sample of the data

        X = X_train.iloc[i*10**6:(i+1)*10**6, :]

        #target variable

        y = y_train[i*10**6:(i+1)*10**6]

        neigh.fit(X,y)

        #take the predictions

        y_test = neigh.predict(X_test)

        #save the predictions

        knnregressoroutputs.append(y_test)

        neigh = 0

        gc.collect()

        

    #average all the predictions

    res = knnregressoroutputs[0]

    for i in knnregressoroutputs[1:]:

        res+=i

    res/=len(knnregressoroutputs)

    return res



def getLassoPredictions(X, y, X_test):

    #kaggle kernel - https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models-latest

    from sklearn import linear_model

    from sklearn.metrics import mean_squared_error

    from math import sqrt

    from tqdm import tqdm

    import matplotlib.pyplot as plt

    

    test_df1 = X.iloc[-10000:,:]

    y_test_actual = y.iloc[-10000:]

    

    X=X.iloc[:len(X)-10000,:]

    y = y.iloc[:len(y)-10000]

    

    #variables needed for grid search

    minrms = float('inf')

    minrmsalpha = -1

    rmserrs = []

    miny = pd.DataFrame()

    #values of alpha

    for i in tqdm(range(0, 5)):

        gc.collect()

        model = linear_model.Lasso(normalize = True, alpha = 10**(-i))

        gc.collect()

        model.fit(X,y)

        y_test = model.predict(X_test)

        

        y_test1 = model.predict(test_df1)

        rms = sqrt(mean_squared_error(y_test_actual, y_test1))

        rmserrs.append(rms)

        del model

        if rms<minrms:

            minrms = rms

            minrmsalpha = i

            miny = y_test

    plt.plot(range(0,5),rmserrs)

    plt.title("Grid search for lasso regression")

    plt.xlabel("alpha (10^-x)")

    plt.ylabel("RMSE")

    return miny



def getRidgePredictions(X, y, X_test):

    #kaggle kernel - https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models-latest

    from sklearn import linear_model

    from sklearn.metrics import mean_squared_error

    from math import sqrt

    from tqdm import tqdm

    import matplotlib.pyplot as plt

    

    test_df1 = X.iloc[-10000:,:]

    y_test_actual = y.iloc[-10000:]

    

    X=X.iloc[:len(X)-10000,:]

    y = y.iloc[:len(y)-10000]

        

    #variables needed for grid search

    minrms = float('inf')

    minrmsalpha = -1

    rmserrs = []

    miny = pd.DataFrame()

    #values of alpha

    for i in tqdm(range(0, 5)):

        gc.collect()

        model = linear_model.Ridge(normalize = True, alpha = 10**(-i))

        gc.collect()

        model.fit(X,y)

        y_test = model.predict(X_test)



        y_test1 = model.predict(test_df1)

        rms = sqrt(mean_squared_error(y_test_actual, y_test1))

        rmserrs.append(rms)

        del model

        if rms<minrms:

            minrms = rms

            minrmsalpha = i

            miny = y_test

    plt.plot(range(0,5),rmserrs)

    plt.title("Grid search for ridge regression")

    plt.xlabel("alpha (10^-x)")

    plt.ylabel("RMSE")

    return miny



def getLRPredictions(X, y, X_test):

    #kaggle kernel - https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models-latest

    from sklearn import linear_model

    #Note: this library uses the closed form expression for the parameters and not gradient descent

    model = linear_model.LinearRegression()

    model.fit(X,y)

    y_test = model.predict(X_test)

    return y_test



def getRFPredictions(X, y, X_test):

    #kaggle kernel: https://www.kaggle.com/pradyu99914/nyc-taxi-fare-models-latest

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=100)

    gc.collect()

    model.fit(X,y)

    y_test = model.predict(X_test)

    return y_test



def getLGBMPredictions(X_train, y_train, X_val, y_val, test_df):

    #get the model

    dtrain = lgb.Dataset(X_train,y_train,silent=False,categorical_feature=['year','month','day','weekday'])

    dval = lgb.Dataset(X_val,y_val,silent=False,categorical_feature=['year','month','day','weekday'])

    model = getLGB()

    pred = model.predict(test_df)

    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':test_df.columns})



    #plt.figure(figsize=(20, 10))

    #sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

    #plt.title('LightGBM Features (avg over folds)')

    #plt.xlabel("Split Gain")

    #plt.tight_layout()

    #plt.show()

    return pred



def getDNNPredictions(X, y, test_df):

    

    #kaggle kernel - https://www.kaggle.com/pradyu99914/fork-of-fork-of-nyc-taxi-fare-models-dl-model?scriptVersionId=23712985

    from keras.models import Sequential

    from keras.layers import Dense,Dropout

    from keras.models import load_model

    try:

        #try to read the model if it is already present

        model = load_model('/kaggle/input/final-pipeline/model.h5')

    except Exception:

        #create and train a new model in case it is not already pesent

        model = Sequential()

        model.add(Dense(2048, input_dim = 28, activation = 'relu'))

        model.add(Dropout(0.2))

        model.add(Dense(1024, activation = 'relu'))

        model.add(Dropout(0.2))

        model.add(Dense(512, activation = 'relu'))

        model.add(Dropout(0.2))

        model.add(Dense(256,  activation = 'tanh'))

        model.add(Dropout(0.2))

        model.add(Dense(128,  activation = 'tanh'))

        model.add(Dense(1, activation = "linear"))

        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

        history = model.fit(X,y, batch_size=2048, epochs = 30)

        import matplotlib.pyplot as plt

        plt.plot(history.history['loss'])

        plt.title('model loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.show()

    #save the model for future commits

    model.save("model.h5")

    y_test = model.predict(test_df)

    return y_test.reshape(len(test_df))



def getXGBpredictions(X_train, y_train, test_df):

    '''import xgboost as xgb

    from bayes_opt import BayesianOptimization

    from sklearn import preprocessing

    from sklearn.metrics import mean_squared_error

    from sklearn.model_selection import train_test_split

    import joblib

    #save model

    try:

        model = joblib.load('/kaggle/input/final-pipeline/xgb.pkl')

    except Exception:

        dtrain = xgb.DMatrix(X_train, label=y_train)

        dtest = xgb.DMatrix(test_df)

        gc.collect()

        def xgb_evaluate(max_depth, gamma, colsample_bytree):

            params = {'eval_metric': 'rmse',

                      'max_depth': int(max_depth),

                      'subsample': 0.8,

                      'eta': 0.1,

                      'gamma': gamma,

                      'verbose_eval':False,

                      'silent':1,

                      'colsample_bytree': colsample_bytree}

            cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    

            return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

        xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (7, 12), 

                                                 'gamma': (0, 1),

                                                 'colsample_bytree': (0.5, 0.9)})

        xgb_bo.maximize(init_points=5, n_iter=10, acq='ei')

        sorted_res = sorted(xgb_bo.res,key = lambda x: x['target'])

        params = sorted_res[-1]

        params['params']['max_depth'] = int(params['params']['max_depth']) 

        model = xgb.train(params, dtrain, num_boost_round=1000, silent = 1)



    joblib.dump(model, "xgb.pkl") 

    # Predict on testing and training set

    y_pred = model.predict(dtest)

    y_train_pred = model.predict(dtrain)

    

    return y_pred'''

    #Please refer to this kernel : https://www.kaggle.com/anushkini/taxi-xgboost

    return None
#getLGBMPredictions(X_train, y_train, X_val, y_val, test_df) #lightgbm

#getkNNpredictions(X_train, y_train, test_df)  #knn - must be atleast 1 million data points for this to work

#getLRPredictions(X_train, y_train, test_df) #linear regression

#getLassoPredictions(X_train, y_train, test_df) #lasso regression

#getRidgePredictions(X_train, y_train, test_df) #ridge regression

#getRFPredictions(X_train, y_train, test_df) random forest regressor

getDNNPredictions(X_train, y_train, test_df) #dnn
'''import requests

import json

from datetime import datetime



#PLEASE TURN INTERNET ON FOR THIS TO WORK... ------->



#read the details

print("Enter the source: ")

place = input()

print("Enter the destination: ")

dest = input()

print("Enter the approximate time in number of hours: ")

time = int(input())

print("Please enter the number of passengers")

psngcnt = int(input())

print("Please enter the date (dd/mm/yyyy)")

date = input().strip()



#coordinates of important places

Manhattan = (-73.9712,40.7831)[::-1]

JFK_airport = (-73.7781,40.6413)[::-1]

Laguardia_airport = (-73.8740,40.7769)[::-1]

statue_of_liberty = (-74.0445,40.6892)[::-1]

central_park = (-73.9654,40.7829)[::-1]

time_square = (-73.9855,40.7580)[::-1]

brooklyn_bridge = (-73.9969,40.7061)[::-1]

rockerfeller = (-73.9787,40.7587)[::-1]



#create a datetime object for the given day

datetime_object = datetime.strptime(date, '%d/%m/%Y')



#perform an api request in order to get the coordinates of the source and destination

response = requests.get("https://api.opencagedata.com/geocode/v1/geojson?q="+place.replace(' ', '+') +"&key=c2f9d990b75444389382e38f107441b0&pretty=1")

srccoords = json.loads(response.text)["features"][0]["geometry"]["coordinates"]

response = requests.get("https://api.opencagedata.com/geocode/v1/geojson?q="+dest.replace(' ', '+') +"&key=c2f9d990b75444389382e38f107441b0&pretty=1")

dstcoords = json.loads(response.text)["features"][0]["geometry"]["coordinates"]



#create a new dataframe for the data point

newdf = pd.DataFrame(columns = test_df.columns)

#create a new row with the extra feature

row = [srccoords[0], 

       srccoords[1],

       dstcoords[0],

       dstcoords[1],

       psngcnt,

       time*60,

       1 if datetime_object.strftime('%d-%m-%y')in us_holidays else 0,

       haversine_distance(srccoords[1], srccoords[0], dstcoords[1], dstcoords[0]),

       datetime_object.year,

       datetime_object.weekday(),

       haversine_distance(Manhattan[0],Manhattan[1],srccoords[1],srccoords[0]),

       haversine_distance(Manhattan[0],Manhattan[1],dstcoords[1],dstcoords[0]),

       haversine_distance(JFK_airport[0],JFK_airport[1],dstcoords[1],dstcoords[0]),

       haversine_distance(JFK_airport[0],JFK_airport[1],srccoords[1],srccoords[0]),

       haversine_distance(Laguardia_airport[0],Laguardia_airport[1],srccoords[1],srccoords[0]),

       haversine_distance(Laguardia_airport[0],Laguardia_airport[1],dstcoords[1],dstcoords[0]),

       datetime_object.day,

       datetime_object.month,

       haversine_distance(statue_of_liberty[0],statue_of_liberty[1],srccoords[1],srccoords[0]),

       haversine_distance(statue_of_liberty[0],statue_of_liberty[1],dstcoords[1],dstcoords[0]),

       haversine_distance(central_park[0],central_park[1],srccoords[1],srccoords[0]),

       haversine_distance(central_park[0],central_park[1],dstcoords[1],dstcoords[0]),

       haversine_distance(time_square[0],time_square[1],srccoords[1],srccoords[0]),

       haversine_distance(time_square[0],time_square[1],dstcoords[1],dstcoords[0]),

       haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],srccoords[1],srccoords[0]),

       haversine_distance(brooklyn_bridge[0],brooklyn_bridge[1],dstcoords[1],dstcoords[0]),

       haversine_distance(rockerfeller[0],rockerfeller[1],srccoords[1],srccoords[0]),

       haversine_distance(rockerfeller[0],rockerfeller[1],dstcoords[1],dstcoords[0])

      ]



#add the row to the dataframe

newdf.loc[len(newdf)] = row

print(newdf)

time*=60



mincost = float('inf')

maxcost = 0

mintime = 0



model = getLGB()



def hours_and_minutes(time):

    hours = (time//60)

    minutes = time - hours*60

    return str(hours)+":"+str(minutes)



#find the best time

for i in range(120):

    newtime = min((max((time-60+i, 0)), 1339)) #taking care of exceptions

    newdf.loc[0, "time"] = newtime

    cost = model.predict(newdf)[0]

    if cost >maxcost:

        maxcost = cost

    if cost < mincost:

        mincost = cost

        mintime = newtime

print("The best time to leave is ", hours_and_minutes(mintime))

print("It will cost you: ", mincost, "USD")

print("savings(best case) in USD:", maxcost-mincost) '''