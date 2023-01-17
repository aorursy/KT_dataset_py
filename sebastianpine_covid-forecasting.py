#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as keb

from scipy.optimize import curve_fit

from datetime import datetime as dtime

from datetime import timedelta



import warnings

warnings.filterwarnings("ignore")
#loading data

path = "../input/input-covid/"

#path = ""

data = pd.read_csv(path+"train2.csv")

data.head()
data["isCountry"] = data.Province_State.isna()

data.Province_State[data.Province_State.isna()] = ""

data["Region"]= data.Country_Region 

data["Region"][~data.isCountry]= data.Country_Region + "-" + data.Province_State

number_of_rows = data.shape[0]

registers_count_by_region = data[["Id", "Region"]].groupby("Region").count()

registers_count_by_region.columns = ["Count"]

registers_count_by_region.reset_index(inplace=True)#

days = np.unique(data.Date)

n_days = len(days)

regions = np.unique(data.Region)



print("Number of rows:", number_of_rows)
registers_count_by_region.head()
print("Days:", days)
print("Countries:", regions)

print("No. Countries:", len(regions))
print("No country - Regions:", data.Region[~data.isCountry])
max_confirmed_per_day_per_country = data[[ "Region", "ConfirmedCases" ]].groupby("Region").max()

max_confirmed_per_day_per_country.head()
#new confirmed column

data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()

data["PreviousConfirmed"] = data["ConfirmedCases"].shift()

data.loc[ data.Id_by_Region==0, "PreviousConfirmed"] = 0

data["NewConfirmed"]= data.ConfirmedCases - data.PreviousConfirmed



#new fatalities column

data["Id_by_Region"] = data[["Region", "Id"]].groupby("Region").cumcount()

data["PreviousFatalities"] = data["Fatalities"].shift()

data.loc[ data.Id_by_Region==0,"PreviousFatalities"] = 0

data["NewFatalities"]= data.Fatalities - data.PreviousFatalities

selected_countries = ["Spain", "Italy", "Germany", "Singapore", 'Korea, South']



fig, ax = plt.subplots(1,2, figsize=(15,5))



for i, country in enumerate(selected_countries):

    

    confirmed_country = data[data.Region==country].ConfirmedCases

    ax[0].plot(days, confirmed_country)

    ax[0].set_xticks(np.arange(0, n_days,20 ))

    

ax[0].legend(selected_countries)

ax[0].set_title("Confirmed cases")

ax[0].grid()



for i, country in enumerate(selected_countries):

    

    fatalities_country = data[data.Region==country].Fatalities

    ax[1].plot(days, fatalities_country)

    ax[1].set_xticks(np.arange(0, n_days,20 ))

    

ax[1].legend(selected_countries)

ax[1].set_title("Fatalities")

ax[1].grid()
fig = plt.figure()



fig, ax = plt.subplots(1,2, figsize=(15,5))



for country in selected_countries:

    

    confirmed_country = np.log(data[data.Region==country].ConfirmedCases+1)

    ax[0].plot(days, confirmed_country)

    ax[0].set_xticks(np.arange(0, n_days,20 ))

ax[0].grid()

ax[0].legend(selected_countries)

ax[0].set_title("Confirmed cases (log)")



for i, country in enumerate(selected_countries):

    

    fatalities_country = np.log(data[data.Region==country].Fatalities+1)

    ax[1].plot(days, fatalities_country)

    ax[1].set_xticks(np.arange(0, n_days,20 ))

    

ax[1].legend(selected_countries)

ax[1].set_title("Fatalities (log)")

ax[1].grid()


fig, ax = plt.subplots(1,2, figsize=(15,5))



for i, country in enumerate(selected_countries):

    

    confirmed_country = data[data.Region==country].NewConfirmed

    ax[0].plot(days, confirmed_country)

    ax[0].set_xticks(np.arange(0, n_days,20 ))

    

ax[0].legend(selected_countries)

ax[0].set_title("Confirmed cases")

ax[0].grid()



for i, country in enumerate(selected_countries):

    

    fatalities_country = data[data.Region==country].NewFatalities

    ax[1].plot(days, fatalities_country)

    ax[1].set_xticks(np.arange(0, n_days,20 ))

    

ax[1].legend(selected_countries)

ax[1].set_title("Fatalities")

ax[1].grid()
data["FirstConfirmed"] = data.apply(lambda x: x["ConfirmedCases"]>0, axis=1)

data["FirstFatality"] = data.apply(lambda x: x["Fatalities"]>0, axis=1)

data["DayFromFirstConfirmed"] = data[["FirstConfirmed", "Region"]].groupby("Region").cumsum()

data["DayFromFirstFatality"] = data[["FirstFatality", "Region"]].groupby("Region").cumsum()



data["TenConfirmed"] = data.apply(lambda x: x["ConfirmedCases"]>10, axis=1)

data["TenFatality"] = data.apply(lambda x: x["Fatalities"]>10, axis=1)

data["DayFromTenConfirmed"] = data[["TenConfirmed", "Region"]].groupby("Region").cumsum()

data["DayFromTenFatality"] = data[["TenFatality", "Region"]].groupby("Region").cumsum()



data["LogNewConfirmed"] = np.log(data["NewConfirmed"]+1) 

data["LogNewFatalities"] = np.log(data["NewFatalities"]+1)
first_day_confirmed = data[data["DayFromTenConfirmed"] == 1][["Region", "Date"]]

first_day_confirmed.head()
first_day_fatality = data[data["DayFromTenFatality"] == 1][["Region", "Date"]]

first_day_fatality.head()
df_merged = pd.merge(first_day_confirmed, first_day_fatality, on='Region', how='left')

print("Number of unique regions:", df_merged.shape[0])

df_merged.head()



def to_datetime(x):

    try: 

        return dtime.strptime(x,  "%Y-%m-%d")

    except: 

        return pd.NaT



df_merged.Date_x = df_merged.Date_x.apply(lambda x: dtime.strptime(x,  "%Y-%m-%d"))

df_merged.Date_y = df_merged.Date_y.apply(lambda x: to_datetime(x))

df_merged["DifferenceFirstConfirmedFatality"] = df_merged.Date_y - df_merged.Date_x

df_merged.head()


fig, ax = plt.subplots(len(selected_countries),1, figsize=(10,10))

error = []

N = 5



def sigmoid(x, a, x0, k):

    y = a / (1 + np.exp(-k*(x-x0)))

    return y



def exponential(x, a, x0, k):

    y = a*np.exp(-k*(x-x0))

    return y



    

for i, country in enumerate(selected_countries):

    

    data_region = data[data.Region == country][data.TenConfirmed==True]



    y = data_region.NewConfirmed

    x = data_region.DayFromTenConfirmed

    ma = np.convolve(y, np.ones((N,))/N, mode='same')



    popt, pcov = curve_fit(sigmoid, x, y)

    confirmed_fitted_sigmoid = sigmoid(x, *popt)

    

    error.append(np.median((confirmed_fitted_sigmoid-ma)**2)/np.max(ma))



    ax[i].plot(x, y)

    ax[i].plot(x, confirmed_fitted_sigmoid)

    ax[i].plot(x, ma)

    ax[i].grid()

error
error_regions = []

for i, country in enumerate(regions):

    

    data_region = data[data.Region == country][data.TenConfirmed==True]







    try:

        y = data_region.NewConfirmed

        x = data_region.DayFromTenConfirmed

        ma = np.convolve(y, np.ones((N,))/N, mode='same')

        popt, pcov = curve_fit(sigmoid, x, y)

        confirmed_fitted_sigmoid = sigmoid(x, *popt)



        error_regions.append(np.median((confirmed_fitted_sigmoid-ma)**2)/np.max(ma))

    except:

        error_regions.append(np.nan)



state_index = pd.DataFrame({"Region": regions, "state_index": error_regions})

df_merged = pd.merge(df_merged, state_index, on='Region', how='left')
interesting_regions = df_merged[df_merged.state_index>20].Region




for i, country in enumerate(interesting_regions):

    

    fig = plt.figure()

    data_region = data[data.Region == country][data.TenConfirmed==True]



    y = data_region.NewConfirmed

    x = data_region.DayFromTenConfirmed

    ma = np.convolve(y, np.ones((N,))/N, mode='same')



    popt, pcov = curve_fit(exponential, x, y)

    confirmed_fitted_sigmoid = exponential(x, *popt)

    

    error.append(np.median((confirmed_fitted_sigmoid-ma)**2)/np.max(ma))



    plt.plot(x, y)

    plt.plot(x, confirmed_fitted_sigmoid)

    plt.plot(x, ma)

    plt.grid()

    plt.title(country)

selected_countries
data_region = data[data.Region == "Korea, South"][data.TenConfirmed==True]
np.median(data_region.NewConfirmed)
error_regions_exponential = []

error_regions_sigmoid = []



for i, country in enumerate(regions):

    

    data_region = data[data.Region == country][data.TenConfirmed==True]





    try:

        y = data_region.NewConfirmed

        x = data_region.DayFromTenConfirmed

        ma = np.convolve(y, np.ones((N,))/N, mode='same')

        



        popt, pcov = curve_fit(sigmoid, x, y)

        confirmed_fitted_sigmoid = sigmoid(x, *popt)

        

        error_regions_sigmoid.append(np.median((confirmed_fitted_sigmoid-ma)**2)/np.max(ma))

    except:

        

        error_regions_sigmoid.append(np.nan)

        



    try:

        y = data_region.NewConfirmed

        x = data_region.DayFromTenConfirmed

        ma = np.convolve(y, np.ones((N,))/N, mode='same')

        



        popt, pcov = curve_fit(sigmoid, x, y)

        confirmed_fitted_exponential = exponential(x, *popt)

        

        error_regions_exponential.append(np.median((confirmed_fitted_exponential-ma)**2)/np.max(ma))

    except:

        

        error_regions_exponential.append(np.nan)

        

        

state_index = pd.DataFrame({"Region": regions, "sigmoid_error": error_regions_sigmoid, 

                            "exponential_error": error_regions_exponential})



df_merged2 = pd.merge(df_merged, state_index, on='Region', how='left')

df_merged2["state"] = df_merged2[["sigmoid_error", "exponential_error"]].apply(lambda x: int(x[0]< x[1]), axis=1)
df_merged2
df_merged2[df_merged2.Region=="Colombia"]
def simple_predictor(func, ts1, ts2, c1, c2, h):

    

    x_past1 = np.arange(len(ts1))

    x_past2 = np.arange(len(ts2))

    

    x_fut = np.arange(c1,h+c1,1)

    

    popt, pcov = curve_fit(func, x_past1, ts1)

    p1 = sigmoid(x_fut, *popt)



    popt, pcov = curve_fit(func, x_past2, ts2)

    p2 = sigmoid(x_fut, *popt)

    

    return p1, p2

    
def multivariate_data(dataset, target, start_index, end_index, history_size,

                      target_size, step, single_step=False):

    data = []

    labels = []



    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size



    for i in range(start_index, end_index):

        indices = range(i-history_size, i, step)

        data.append(dataset[indices])



        if single_step:

            labels.append(target[i+target_size])

        else:

            labels.append(target[i:i+target_size])



    return np.array(data), np.array(labels)





def RMSLE (y_pred,  y_true):

    

    return keb.mean(keb.square(keb.log(y_pred+1)-keb.log(y_true+1)))

    

datasets = []

norm_factor = 100



for region in regions:

    datasets.append(np.array(data[["NewConfirmed", "NewFatalities"]][data.TenConfirmed==1][data.Region==region]))

    

len_data = len(datasets)
past_history = 7

future_target = 1

STEP = 1



BATCH_SIZE = 10

BUFFER_SIZE = 10

EPOCHS = 0

EVALUATION_INTERVAL = 1

THRESHOLD_REGION = 15



x_train = []

y_train = []

x_val = []

y_val = []



for dataset in datasets:

    

    if (dataset.shape[0]>THRESHOLD_REGION):

        len_data= dataset.shape[0]

        TRAIN_SPLIT = int(len_data*0.6)

        x_train_temp, y_train_temp = multivariate_data(dataset, dataset[:,0], 0, TRAIN_SPLIT, past_history, future_target, STEP)



        x_train.append(x_train_temp)

        y_train.append(y_train_temp)

        

        x_val_temp, y_val_temp = multivariate_data(dataset, dataset[:,0], TRAIN_SPLIT, None, past_history, future_target, STEP)

        

        if(x_val_temp.shape[0]>0):

            x_val.append(x_val_temp)

            y_val.append(y_val_temp)

            
x_train = np.concatenate(x_train, axis=0)

y_train = np.vstack(y_train)





print("x_train shape:", x_train.shape)

print("y_train shape:", y_train.shape)



x_val = np.concatenate(x_val, axis=0)

y_val = np.vstack(y_val)





print("x_train shape:", x_val.shape)

print("y_train shape:", y_val.shape)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

val_data = val_data.batch(BATCH_SIZE).repeat()
multi_step_model = tf.keras.models.Sequential()

multi_step_model.add(tf.keras.layers.LSTM(16,

                                          return_sequences=True,

                                          input_shape=x_train.shape[-2:]))

multi_step_model.add(tf.keras.layers.LSTM(8, activation='relu'))

multi_step_model.add(tf.keras.layers.Dense(future_target))



multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001, clipvalue=1.0), loss=RMSLE)
multi_step_history = multi_step_model.fit(train_data, epochs=EPOCHS,

                                          steps_per_epoch=EVALUATION_INTERVAL,

                                          validation_data=val_data,

                                          validation_steps=10)
pred= multi_step_model.predict(x_val[1:100])
plt.scatter(pred, y_val[1:100])
path = "../input/covid19-global-forecasting-week-3/"

test_data = pd.read_csv(path+"test.csv")

test_data.head()
test_data["isCountry"] = test_data.Province_State.isna()

test_data.Province_State[test_data.Province_State.isna()] = ""

test_data["Region"]= test_data.Country_Region 

test_data["Region"][~test_data.isCountry]= test_data.Country_Region + "-" + test_data.Province_State



number_of_rows = test_data.shape[0]

registers_count_by_region_test = test_data[["ForecastId", "Region"]].groupby("Region").count()

registers_count_by_region_test.columns = ["Count"]

registers_count_by_region_test.reset_index(inplace=True)#

days_test = np.unique(test_data.Date)

n_days_test = len(days_test)
test_data
days_test
days_to_predict = [dtime.strftime(dtime.strptime(days[-1],  "%Y-%m-%d")+timedelta(i), "%Y-%m-%d") for i in range(1,31)]

n_days_to_predict = len(days_to_predict)

concat_df = pd.DataFrame()



for i, region in enumerate(regions):

    

    data_region = data[data.Region == region][data.TenConfirmed==True]

    

    if (data_region.shape[0]==0):

        data_region = data[data.Region == region][data.FirstConfirmed==True]

        

    try:

        state = df_merged2[df_merged2.Region==region].state.iloc[0]

    except:

        state = 0

        

    

        

    temp_df = pd.DataFrame({"Date":days_to_predict, "Country_Region": [data_region.Country_Region.iloc[0]]*n_days_to_predict, 

                            "Province_State": [data_region.Province_State.iloc[0]]*n_days_to_predict,

                            "isCountry": [data_region.isCountry.iloc[0]]*n_days_to_predict})



    temp_df["Region"]= temp_df.Country_Region 

    temp_df["Region"][~temp_df.isCountry] = temp_df.Country_Region + "-" + temp_df.Province_State

    

    ts_confirmed = data_region.NewConfirmed

    ts_fatalities = data_region.NewFatalities

    

    ma_conf = np.convolve(ts_confirmed, np.ones((N,))/N, mode='same')

    ma_fat = np.convolve(ts_fatalities, np.ones((N,))/N, mode='same')

    

    current_day_confirmed = data_region.DayFromTenConfirmed.iloc[-1]

    current_day_fatalities = data_region.DayFromTenFatality.iloc[-1]

    

    current_confirmed = data_region.ConfirmedCases.iloc[-1]

    current_fatalities = data_region.Fatalities.iloc[-1]

        



    func = sigmoid



    

    try:

        predicted_confirmed, predicted_fatalities = simple_predictor(func, ma_conf, ma_fat, current_day_confirmed, current_day_fatalities, n_days_to_predict)

        

        predicted_confirmed[0] = predicted_confirmed[0] + current_confirmed

        predicted_confirmed = np.cumsum(predicted_confirmed)

        

        predicted_fatalities[0] = predicted_fatalities[0] + current_fatalities

        predicted_fatalities = np.cumsum(predicted_fatalities)

    

    except:

        

        predicted_confirmed = [0]*n_days_to_predict

        predicted_fatalities = [0]*n_days_to_predict

        predicted_confirmed[0], predicted_fatalities[0] = current_confirmed, current_fatalities

        predicted_confirmed = np.cumsum(predicted_confirmed)

        predicted_fatalities = np.cumsum(predicted_fatalities)

    

    temp_df["ConfirmedCases"] = predicted_confirmed

    temp_df["Fatalities"] = predicted_fatalities

    concat_df = pd.concat([concat_df, temp_df])
current_data = data[["Date", "Country_Region", "Province_State", "ConfirmedCases", "Fatalities", "Region"]]

concat_df
submission1 = test_data.merge(current_data, on=["Region", "Date"] ,  how='left')

submission1 
submission2 = submission1.merge(concat_df, on= ["Region", "Date"], how="left")
submission2[submission2.Region=="Spain"].head()
submission2["ConfirmedCases"] = submission2[["ConfirmedCases_x", "ConfirmedCases_y"]].apply(lambda x: x[0] if ~np.isnan(x[0]) else x[1], axis=1)

submission2["Fatalities"] = submission2[["Fatalities_x", "Fatalities_y"]].apply(lambda x: x[0] if ~np.isnan(x[0]) else x[1], axis=1)
submission_data = pd.read_csv(path+"submission.csv")#

submission_data
submission3 = submission_data[["ForecastId"]].merge(submission2[["ForecastId", "ConfirmedCases", "Fatalities" ]], on= "ForecastId", how="left")



submission3 = submission3.astype("int32")
submission3.to_csv("submission.csv", index=False)