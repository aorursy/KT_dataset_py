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
import pandas as pd



# df = import_data("/kaggle/input/crimes-in-boston/crime.csv")

df = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv", engine = 'python')
df.head()
df = df[['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE','YEAR','MONTH','DAY_OF_WEEK','Lat','Long']]
df = df.dropna()
df.reset_index(inplace=True)
# Filtering the locations to get all coordinates in the Boston area



lis = []



Lat = list(df['Lat'])

Long = list(df['Long'])



for i in range(len(Lat)):

    li = []

    if Lat[i] > 30 and Long[i] < -40:

        li.append(Lat[i])

        li.append(Long[i])

    else:

        li.append(np.nan)

        li.append(np.nan)

    lis.append(li)



del Lat

del Long



L = np.array(lis)

Latitude = L[:, 0]

Longitude = L[:, 1]



df['Latitude'] = Latitude

df['Longitude'] = Longitude



df = df.dropna()

df.reset_index(inplace = True)

df.head()
df = df[['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE','YEAR','MONTH','DAY_OF_WEEK','Latitude','Longitude']]



Latitude = list(df['Latitude'])

Longitude = list(df['Longitude'])



L = [[Latitude[i], Longitude[i]] for i in range(len(Latitude))]



X = np.array(L)

X.shape
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



kmeans = KMeans(n_clusters = 10, random_state = 0).fit(X)



Clusters = kmeans.labels_.tolist()



# Plot the cluster assignments



plt.scatter(X[:, 0], X[:, 1], c = Clusters, cmap = "plasma")

plt.xlabel("Latitude")

plt.ylabel("Longitude")



# For a specific cluster, use X_ (from the above cell) instead of X
df['Cluster'] = Clusters

df.head()
df['OCCURRED_ON_DATE'] = pd.to_datetime(df.OCCURRED_ON_DATE)
df = df.sort_values(by='OCCURRED_ON_DATE')

df.reset_index(inplace = True)
lisp = [i.date() for i in list(df['OCCURRED_ON_DATE'])]



df['OCCURRED_ON_DATE'] = lisp
def initialize_frames(df, set_of_clusters):

    frame_cache = {}

    

    for i in range(len(set_of_clusters)):

        df_x = df[df['Cluster'] == set_of_clusters[i]]

        frame_cache['df' + str(i)] = df_x

    

    return frame_cache



set_of_clusters = list(set(Clusters))

frame_cache = initialize_frames(df, set_of_clusters)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt



def get_metrics(y_actual, y_predicted, algorithm = '', model_number = '', save_to_file = False):

    if save_to_file:

        if algorithm == '' or algorithm is None:

            print("Please enter algorithm name for the metrics file")

            return

    

        f = open('KMEANS_' + algorithm + '_metrics.txt', "a+")

    

        if model_number == '' or model_number is None:

            print("Please enter model number")

            return

        

        f.write("\n\n***************** Model " + model_number + " *******************\n\n")

        f.write("Root mean squared error (RMSE) => " + str(sqrt(mean_squared_error(y_actual, y_predicted))) + "\n")

        f.write("Mean squared error (MSE) => " + str(mean_squared_error(y_actual, y_predicted)) + "\n")

        f.write("Mean abolute error (MAE) => " + str(mean_absolute_error(y_actual, y_predicted)) + "\n")

    else:

        print("***************** Prediction Metrics *******************\n\n")

        print("Root mean squared error (RMSE) => " , sqrt(mean_squared_error(y_actual, y_predicted)))

        print("Mean squared error (MSE) => " , mean_squared_error(y_actual, y_predicted))

        print("Mean absolute error (MAE) => ", mean_absolute_error(y_actual, y_predicted))

        

    print("Model_" + model_number + "'s metrics have been recorded")
import pickle



def save_model(model, model_number, model_type, last_date):

    

    pkl_path = ''

    if model_type == 'Prophet':

        pkl_path = 'KMEANS_' + model_type + '_' + model_number + '.pkl'

    elif model_type == 'ETS' or model_type == 'ARIMA':

        if last_date is None or last_date == '':

            print("Please enter the last training date for the model")

            return

        if model_number is None or model_number == '':

            print("Please enter the model number")

            return

        pkl_path = 'KMEANS_' + model_type + '_' + model_number + '_' + last_date + '.pkl'

    else:

        print("Please enter the correct model type: (Prophet, ETS or ARIMA)")

        return

    

    with open(pkl_path, 'wb') as f:

        pickle.dump(model, f)



def load_model(pkl_path):

    model = None

    try:

        with open(pkl_path, 'rb') as f:

            model = pickle.load(f)

    except pickle.UnpicklingError:

        with open(pkl_path, 'rb') as f:

            model = f.read()

    return model
# Forecasting for the test data using Prophet



def ProphetsForecast(model, no_of_days):

    

    future = model.make_future_dataframe(periods = no_of_days)

    forecast = model.predict(future)

    return forecast



def TestProphet(model, df_train, df_test):

    actual_dates = list(df_test['ds'])

    LatestDate = actual_dates[-1]

    LastDate = list(df_train['ds'])[-1]

    

    no_of_days = (LatestDate - LastDate).days

    if no_of_days <= 0:

        print("Please enter a date after " + str(LastDate))

        return

    

    forecast = ProphetsForecast(model, no_of_days)

    yhat = list(forecast['yhat'])

    

    predictions = list()

    for DATE in actual_dates:

        diff = (DATE - LastDate).days

        predictions.append(yhat[diff - 1])

        

    return predictions
# Creating Prophet models



import datetime

from fbprophet import Prophet



def createProphet(x, df_x, save_metrics = False, save_Model = False):

    

    df_x.reset_index(inplace = True)

    df_x = df_x.groupby(['OCCURRED_ON_DATE']).count()[['level_0','index']]

    df_x.drop(columns = ['index'], inplace = True)

    

    df_train = df_x[0 : int(len(df_x) * 0.8)]

    df_train = df_train.reset_index()

    df_train.rename(columns = {'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)

    

    df_test = df_x[int(len(df_x) * 0.8) : ]

    df_test = df_test.reset_index()

    df_test.rename(columns = {'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)

    

    # Prophet models cannot make predictions using just 1 training sample

    

    while len(df_train) < 2:

        train_date = list(df_train['ds'])

        train_y = list(df_train['y'])

        

        test_date = list(df_test['ds'])

        test_y = list(df_test['y'])

        

        train_date.append(test_date[0])

        train_y.append(test_y[0])

        

        test_date.remove(test_date[0])

        test_y.remove(test_y[0])

        

        if len(test_date) > 0:

            Next_Date = test_date[-1] + datetime.timedelta(days = 1)

            if len(test_y) == 0:

                Next_Y = 0

            else:

                Next_Y = sum(test_y) // len(test_y)

            test_date.append(Next_Date)

            test_y.append(Next_Y)

        else:

            Next_Date = train_date[-1] + datetime.timedelta(days = 1)

            if len(train_y) == 0:

                Next_Y = 0

            else:

                Next_Y = sum(train_y) // len(train_y)

            test_date.append(Next_Date)

            test_y.append(Next_Y)

        

        df_train = pd.DataFrame()

        df_test = pd.DataFrame()

        

        df_train['ds'] = train_date

        df_test['ds'] = test_date

        

        df_train['y'] = train_y

        df_test['y'] = test_y

    

    prophet = Prophet()

    prophet.fit(df_train)

    

    predicted = TestProphet(prophet, df_train, df_test)

    

    get_metrics(df_test['y'], predicted, 'Prophet', str(x), save_metrics)

    

    if save_Model:

        save_model(prophet, str(x), 'Prophet', last_date = '')
# Note: While creating the models, always initialize the frame_cache before calling the create functions for any model

frame_cache = initialize_frames(df, set_of_clusters)

for i in range(len(set_of_clusters)):

    createProphet(i, frame_cache['df' + str(i)], save_metrics = False, save_Model = False)

    

# Metrics are stored in DBSCAN_Prophet_metrics.txt
# Forecasting for the test data using ETS



from statsmodels.tsa.holtwinters import ExponentialSmoothing



def ETSForecast(model, no_of_days):

    

    test_predictions = model.forecast(no_of_days).rename('TES Forecast')

    return list(test_predictions)



def TestETS(model, df_train, df_test):

    actual_dates = list(df_test['ds'])

    LatestDate = actual_dates[-1]

    LastDate = list(df_train['ds'])[-1]

    

    no_of_days = (LatestDate - LastDate).days

    if no_of_days <= 0:

        print("Please enter a date after " + str(LastDate))

        return

    

    predicted = ETSForecast(model, no_of_days)

#     print(len(predicted))

    

    predictions = list()

    for DATE in actual_dates:

        diff = (DATE - LastDate).days

#         print(diff)

        predictions.append(predicted[diff - 1])

        

    return predictions
# ETS



import datetime

from statsmodels.tsa.holtwinters import ExponentialSmoothing



def createETS(x, df_x, save_metrics = False, save_Model = False):

    df_x.reset_index(inplace = True)

    df_x = df_x.groupby(['OCCURRED_ON_DATE']).count()[['level_0','index']]

    df_x.drop(columns=['index'], inplace = True)

    

    df_train = df_x[0 : int(len(df_x) * 0.8)]

    df_train = df_train.reset_index()

    df_train.rename(columns = {'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)

    

    df_test = df_x[int(len(df_x) * 0.8) : ]

    df_test = df_test.reset_index()

    df_test.rename(columns = {'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)

    

    while len(df_train) < 2:

        train_date = list(df_train['ds'])

        train_y = list(df_train['y'])

        

        test_date = list(df_test['ds'])

        test_y = list(df_test['y'])

        

        train_date.append(test_date[0])

        train_y.append(test_y[0])

        

        test_date.remove(test_date[0])

        test_y.remove(test_y[0])

        

        if len(test_date) > 0:

            Next_Date = test_date[-1] + datetime.timedelta(days = 1)

            if len(test_y) == 0:

                Next_Y = 0

            else:

                Next_Y = sum(test_y) // len(test_y)

            test_date.append(Next_Date)

            test_y.append(Next_Y)

        else:

            Next_Date = train_date[-1] + datetime.timedelta(days = 1)

            if len(train_y) == 0:

                Next_Y = 0

            else:

                Next_Y = sum(train_y) // len(train_y)

            test_date.append(Next_Date)

            test_y.append(Next_Y)

        

        df_train = pd.DataFrame()

        df_test = pd.DataFrame()

        

        df_train['ds'] = train_date

        df_test['ds'] = test_date

        

        df_train['y'] = train_y

        df_test['y'] = test_y

    

    # For some cluster organizations, Triple Exponential Smoothing may not work

    

    s_p = 2

    ets_model = ExponentialSmoothing(df_train['y']).fit()



#     triple_model = ExponentialSmoothing(df_train['y'], trend = 'add', seasonal = 'add', seasonal_periods = s_p).fit()

    test_predictions = TestETS(ets_model, df_train, df_test)

    

    get_metrics(df_test['y'], test_predictions, 'ETS', str(x), save_metrics)

    

    if save_Model:

        save_model(ets_model, str(x), 'ETS', str(list(df_train['ds'])[-1]))
frame_cache = initialize_frames(df, set_of_clusters)

for i in range(len(set_of_clusters)):

    createETS(i, frame_cache['df' + str(i)], save_metrics = False, save_Model = False)
# Prepare frames for ARIMA model



import datetime



def initialize_ARIMA_frames(df, Clusters):

    frame_ARIMA_cache = {}

    

    for i in range(len(Clusters)):

        df_x = df[df['Cluster'] == Clusters[i]]

        df_x.reset_index(inplace = True)

        df_x = df_x.groupby(['OCCURRED_ON_DATE']).count()[['level_0','index']]

        df_x.drop(columns = ['index'], inplace = True)

        df_x.reset_index(inplace = True)

        df_x.rename(columns = {'OCCURRED_ON_DATE':'ds', 'level_0':'y'} , inplace = True)

        frame_ARIMA_cache['df' + str(i)] = df_x

    

    return frame_ARIMA_cache



frame_ARIMA_cache = initialize_ARIMA_frames(df, set_of_clusters)



def prepare_data(frame_ARIMA_cache, set_of_clusters):

    for i in range(len(set_of_clusters)):

        df_x = frame_ARIMA_cache['df' + str(i)]

        df_date = list(df_x['ds'])

        df_y = list(df_x['y'])

    

        new_dates = [df_date[0]]

        new_y = [df_y[0]]

    

        for I in range(1, len(df_date)):

            no_of_days = (df_date[I] - df_date[I - 1]).days

            starting_date = df_date[I - 1]

        

            while no_of_days > 1:

                next_date = starting_date + datetime.timedelta(days = 1)

                next_y = 0

            

                new_dates.append(next_date)

                new_y.append(next_y)

            

                no_of_days -= 1

                starting_date = next_date

            

            new_dates.append(df_date[I])

            new_y.append(df_y[I])

        

        df_x = pd.DataFrame()

        df_x['ds'] = new_dates

        df_x['y'] = new_y

    

        frame_ARIMA_cache['df' + str(i)] = df_x

    return frame_ARIMA_cache



frame_ARIMA_cache = prepare_data(frame_ARIMA_cache, set_of_clusters)
# Forecasting for the test data using ARIMA



from statsmodels.tsa.arima_model import ARIMA



def ARIMAForecast(model, no_of_days):

    forecast = model.forecast(no_of_days)[0]

    return forecast



def TestARIMA(model, df_train, df_test):

    actual_dates = list(df_test['ds'])

    LatestDate = actual_dates[-1]

    LastDate = list(df_train['ds'])[-1]

    

    no_of_days = (LatestDate - LastDate).days

    if no_of_days <= 0:

        print("Please enter a date after " + str(LastDate))

        return

    

    predicted = ARIMAForecast(model, no_of_days)

#     print(len(predicted))

    

    predictions = list()

    for DATE in actual_dates:

        diff = (DATE - LastDate).days

#         print(diff)

        predictions.append(predicted[diff - 1])

        

    return predictions
# ARIMA



from statsmodels.tsa.arima_model import ARIMA



def createARIMA(x, df_x, save_metrics = False, save_Model = False):

    

    df_train = df_x[0 : int(len(df_x) * 0.8)]

    

    df_test = df_x[int(len(df_x) * 0.8) : ]

    

    model = ARIMA(np.asarray(df_train['y']), order = (1, 1, 0))

    model_fit = model.fit(disp = 0)

    predicted = TestARIMA(model_fit, df_train, df_test)

    

    get_metrics(df_test['y'], predicted, 'ARIMA', str(x), save_metrics)

    

    if save_Model:

        save_model(model_fit, str(x), 'ARIMA', last_date = str(list(df_train['ds'])[-1]))
for i in range(len(set_of_clusters)):

    createARIMA(str(i), frame_ARIMA_cache['df' + str(i)], save_metrics = False, save_Model = False)
import os



def delete_output_files(specific_file = ''):

    if specific_file == '' or specific_file is None:

        output_files = os.listdir('/kaggle/working')

    

        for i in output_files:

            os.remove(i)

    else:

        os.remove(specific_file)

        

# un-comment the next line this cell after you have downloaded the files you need

# delete_output_files()
# Testing out the models for a given cluster



import os

import datetime



# input_date = input("Enter date (dd-mm-YYYY): ")

# input_date = datetime.strptime(input_date, '%d-%m-%Y')

# cl = int(input("Enter the cluster for which you want to test: "))



input_date = datetime.datetime.now()

cl = 0



# Create the models



frame_cache = initialize_frames(df, set_of_clusters)

createProphet(cl, frame_cache['df' + str(cl)], save_metrics = False, save_Model = True)



frame_cache = initialize_frames(df, set_of_clusters)

createETS(cl, frame_cache['df' + str(cl)], save_metrics = False, save_Model = True)



createARIMA(cl, frame_ARIMA_cache['df' + str(cl)], save_metrics = False, save_Model = True)



output_list = os.listdir()



# Load the models



prophet = ets = arima = None

pkl_prophet = pkl_ets = pkl_arima = ''



for i in output_list:

    if 'Prophet' in i and not 'metrics' in i:

        pkl_prophet = i

    elif 'ETS' in i and not 'metrics' in i:

        pkl_ets = i

    elif 'ARIMA' in i and not 'metrics' in i:

        pkl_arima = i

        

prophet = load_model(pkl_prophet)

ets = load_model(pkl_ets)

arima = load_model(pkl_arima)



# Testing prophet



w_prophet = None

last_date = list(prophet.history_dates)[-1]

no_of_days = (input_date - last_date).days



if no_of_days > 0:

    

    forecast = ProphetsForecast(prophet, no_of_days)

    w_prophet = list(forecast['yhat'])[-1]

    print("Prophet forecasts: " + str(w_prophet))

else:

    print("Please enter a date which is after " + str(last_date))

    

# Testing ETS



w_ets = None

last_str = pkl_ets.split('_')[-1].split('.')[0]

last_date = datetime.datetime.strptime(last_str, '%Y-%m-%d')



no_of_days = (input_date - last_date).days



if no_of_days > 0:

    predictions = ets.forecast(no_of_days).rename('TES Forecast')

    w_ets = list(predictions)[-1]

    print("ETS forecasts: " + str(w_ets))

else:

    print("Please enter a date which is after " + str(last_date))

    

# Testing ARIMA



w_arima = None

last_str = pkl_arima.split('_')[-1].split('.')[0]

last_date = datetime.datetime.strptime(last_str, '%Y-%m-%d')



no_of_days = (input_date - last_date).days



if no_of_days > 0:

    forecast = arima.forecast(no_of_days)[0]

    w_arima = forecast[-1]

    print("ARIMA forecasts: " + str(w_arima))

else:

    print("Please enter a date which is after " + str(last_date))



delete_output_files(specific_file = 'KMEANS_Prophet_0.pkl')

delete_output_files(specific_file = 'KMEANS_ETS_0_2018-01-10.pkl')

delete_output_files(specific_file = 'KMEANS_ARIMA_0_2018-01-10.pkl')