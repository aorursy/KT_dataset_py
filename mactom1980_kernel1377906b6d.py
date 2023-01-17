# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

files = [None]*3
i=0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        files[i] = os.path.join(dirname, filename)
        i = i + 1
# Any results you write to the current directory are saved as output.
for i in range(3):
    f = files[i].find('train')
    if f!=-1:
        train_dt = pd.read_csv(files[i])
for i in range(3):
    f = files[i].find('test')
    if f!=-1:
        test_dt = pd.read_csv(files[i])
for i in range(3):
    f = files[i].find('submission')
    if f!=-1:
        submi_dt = pd.read_csv(files[i])
end_train = train_dt.shape[0]
from pandas.api.types import CategoricalDtype

# concatenate train_and test sets

test_train_dt = train_dt.append(test_dt, sort=False)

# test_train_dt[19698-5:19698+5]
# test_train_dt.shape

# make Date Categorical
uniqueDates = list(test_train_dt.Date.unique())
cat_type_date = CategoricalDtype(categories = uniqueDates , ordered=True)
# test_train_dt.Date = 
test_train_dt.Date = test_train_dt.Date.astype(cat_type_date).cat.codes.astype(float)
# Province_State categorical NaN = -1.0
test_train_dt.Province_State = test_train_dt.Province_State.astype("category").cat.codes.astype(float)
# Country_Region categorical
test_train_dt.Country_Region = test_train_dt.Country_Region.astype("category").cat.codes.astype(float)
test_train_dt
# test_train_dt['SinceTheOutbreak'] = 0
# test_train_dt

# nrOfRows = test_train_dt.shape[0]
# # nrOfRows

# sinceTheOutbreak = np.zeros((nrOfRows,1))
# # for i in range(nrOfRows):

# test_train_dt[test_train_dt['Country_Region']==0]
# sinceTheOutbreak[test_train_dt['Country_Region']==0,:] = 0
# test_train_dt['Country_Region']==0 & test_train_dt['Country_region']==0
test_train_dt
train_set = test_train_dt[:end_train]
test_set = test_train_dt[end_train:]

time_span = 7;
nuberOfFeatures = 2*time_span +1

train_set.shape[0]
nrOfTrainDates = train_set.Date.unique().shape[0]
# find the number of countries without provices

nrOfCountriesNoRegion = train_set[train_set.Province_State==-1].Country_Region.unique().shape[0]

# find the number of countries with provinces

nrOfTrainProvinces = train_set[train_set.Province_State!=-1].Province_State.unique().shape[0]
# find the number of different countries/states

nrOfCountries_ALL = nrOfTrainProvinces + nrOfCountriesNoRegion
# nrOfCountries_ALL
nrOfTrainData = nrOfCountries_ALL*(nrOfTrainDates-time_span) #+1
# nrOfTrainData
X = np.zeros((nrOfTrainData,nuberOfFeatures))
Y = np.zeros((nrOfTrainData,2))
# X.shape
# calcuate the labels for the train set

inst_count = 0;
#max country ID + 1
max_country_ID = train_set.Country_Region.unique().max()+1

train_set['Province_State'] = train_set['Province_State']+max_country_ID
test_set['Province_State'] = test_set['Province_State']+max_country_ID
# fix greenland



# train_set[train_set["Province_State"]>-1] + max_country_ID

for i in range(nrOfCountriesNoRegion):



    country = train_set[train_set['Country_Region']==i]
    if country.Province_State.unique().shape[0] == 1:

        nrOfdaysPerCountry = country.shape[0]
        for j in range(nrOfdaysPerCountry-time_span):
            X[inst_count:inst_count+1, 0:time_span] = country.ConfirmedCases[j:time_span+j].values
            X[inst_count:inst_count+1, time_span:2*time_span] = country.Fatalities[j:time_span+j].values
            X[inst_count:inst_count+1, 2*time_span:2*time_span+1] = country.Country_Region[j:j+1].values
            Y[inst_count:inst_count+1,0:1] = country.ConfirmedCases[time_span+j:time_span+j+1].values
            Y[inst_count:inst_count+1,1:2] = country.Fatalities[time_span+j:time_span+j+1].values
    #         print(country.Date[time_span+j:time_span+j+1].values)
            inst_count = inst_count + 1

    else:
        nrOfProvinces = country.Province_State.unique().shape[0]
    #     print(nrOfProvinces)
    #     print(country.Province_State.unique())
        for ii in country.Province_State.unique():
    #         train_set[train_set['Country_Region']==i]
            province = country[country['Province_State']==ii]

            nrOfdaysPerProvince = province.shape[0]
            for j in range(nrOfdaysPerProvince-time_span):
                print(j)
                X[inst_count:inst_count+1, 0:time_span] = province.ConfirmedCases[j:time_span+j].values
                X[inst_count:inst_count+1, time_span:2*time_span] = province.Fatalities[j:time_span+j].values
                X[inst_count:inst_count+1, 2*time_span:2*time_span+1] = province.Province_State[j:j+1].values
                Y[inst_count:inst_count+1,0:1] = province.ConfirmedCases[time_span+j:time_span+j+1].values
                Y[inst_count:inst_count+1,1:2] = province.Fatalities[time_span+j:time_span+j+1].values
        #         print(country.Date[time_span+j:time_span+j+1].values)
                inst_count = inst_count + 1





# from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.optimizers import rmsprop
from keras.constraints import nonneg

import numpy as np
import numpy.random as nr
# from tensorflow import set_random_seed
import matplotlib.pyplot as plt


from keras.layers import Dropout, LeakyReLU
def plot_loss(history):
    train_loss = history.history['loss']
#     test_loss = history.history['val_loss']
    x = list(range(1, len(train_loss) + 1))
#     plt.plot(x, test_loss, color = 'red', label = 'test loss')
    plt.plot(x, train_loss, label = 'traning loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

# kernel_regularizer=regularizers.l2(0.1)
nn = models.Sequential()
nn.add(layers.Dense(128, activation = 'linear', input_shape = (nuberOfFeatures, ), 
                    kernel_regularizer=regularizers.l2(0.01) )) #,
#                     kernel_constraint=nonneg()))
# nn.add(Dropout(rate = 0.2))
nn.add(layers.Dense(64, activation = 'linear',
                    kernel_regularizer=regularizers.l2(0.01) )) #,
#                     kernel_constraint=nonneg()))
# nn.add(Dropout(rate = 0.2))
nn.add(layers.Dense(32, activation = 'relu', 
                    kernel_regularizer=regularizers.l2(0.01) )) #,
#                     kernel_constraint=nonneg()))
# nn.add(Dropout(rate = 0.2))

nn.add(layers.Dense(2, activation = 'linear'))
nn.summary()

# kernel_regularizer=regularizers.l1(10.0)


nn.compile(optimizer = 'rmsprop', loss = 'mean_squared_logarithmic_error', 
                metrics = ['mean_squared_logarithmic_error'])
history = nn.fit(X, Y, 
                  epochs = 200, batch_size = nrOfdaysPerCountry, verbose = 0) #,validation_data = (X, Y))



plot_loss(history)



nrOfTestInst = test_set.shape[0]

X_test = np.zeros((nrOfTestInst, nuberOfFeatures))

# the train dates
testDates = test_set.Date.unique()
firstTestDay = testDates.min()
for i in range(nrOfTestInst):
    # i=3999
    # starts from 0!!!!!!
    # i=353
    country_index = test_set.Country_Region[i:i+1].values[0]
    province_index = test_set.Province_State[i:i+1].values[0]
    # province_index = 172 means NaN
    # is a country or region
    # test_set['Country_Region']==0

    test_instance = test_set[i:i+1]
    test_date = test_instance.Date.values[0]


    # if test_set[test_set['Country_Region'] == country_index].Province_State.unique().shape[0] == 1:
    # print('# is a country')
    # how many days we need
    start_previous_days = test_date - time_span
    end_previous_days = test_date - 1
    # how many days from test set


    only_test_set_day = test_date-firstTestDay
    if only_test_set_day>=time_span:
        test_set_days = time_span
    else:
        test_set_days = test_date - firstTestDay   
    # how many days from the train set
    train_set_days = time_span - test_set_days



    #from train set 
    #     print(train_set[train_set['Country_Region']==country_index and train_set[Date]>=start_previous_days])
    #     train_set['Country_Region']==country_index and train_set['Date']>=start_previous_days
    # from train set
    if test_set[test_set['Country_Region'] == country_index].Province_State.unique().shape[0] == 1:
        tmp_count = train_set[train_set['Country_Region']==country_index]
    else:
    #     print('state')
        if province_index != 172:
            tmp_count = train_set[train_set['Province_State']==province_index]
        else:
            tmp_count_0 = train_set[train_set['Country_Region']==country_index]
            tmp_count = tmp_count_0[tmp_count_0['Province_State']==province_index]
    tmp_count1 = tmp_count[tmp_count['Date']>=start_previous_days]
    tmp_train_set_days = tmp_count1[tmp_count1['Date']<=start_previous_days+train_set_days-1]
    # print('train')
    # print(tmp_train_set_days)
    # from test set
    test_set_start_day = test_date - test_set_days
    test_set_end_day = test_date - 1
    #     print(test_set_end_day)
    if test_set[test_set['Country_Region'] == country_index].Province_State.unique().shape[0] == 1:
        tmp_test = test_set[test_set.Country_Region==country_index]
    else:
    #     print('state')
        if province_index != 172:
            tmp_test = test_set[test_set.Province_State==province_index]
        else:
    #         tmp_test_0 = test_set[test_set.Province_State==province_index]
            tmp_test_0 = test_set[test_set.Country_Region==country_index]
            tmp_test = tmp_test_0[tmp_test_0.Province_State==province_index]

    tmp_test1 = tmp_test[tmp_test.Date>=test_set_start_day]
    tmp_test_set_days = tmp_test1[tmp_test1.Date<=test_set_end_day]
    # print('test')
    # print(tmp_test_set_days)
    features_df = tmp_train_set_days.append(tmp_test_set_days)
    #     print(features_df)
    X_test[i:i+1, 0:time_span] = features_df.ConfirmedCases[:].values
    X_test[i:i+1, time_span:2*time_span] = features_df.Fatalities[:].values
    X_test[i:i+1, 2*time_span:2*time_span+1] = features_df.Country_Region[0:1].values
    prediction = nn.predict(X_test[i:i+1])
    #     print(prediction)
#     if prediction[0,0] < 0:
#         prediction[0,0] = 0
#     if prediction[0,1] < 0:
#         prediction[0,1] = 0
    test_set.set_value(i, 'ConfirmedCases', round(prediction[0,0]))
    test_set.set_value(i, 'Fatalities',round(prediction[0,1]))

# find the overlap between test and train
country_index = 0
trainDates = train_set[train_set['Country_Region']==0].Date
testDates = test_set[test_set['Country_Region']==0].Date
common_Dates = np.intersect1d(trainDates, testDates)

nrOfCommon_Dates = common_Dates.shape[0]

S = 0
n = 0

for i in range(nrOfCommon_Dates):
    # common_Dates[i]
    tmp_train = train_set[train_set['Date']==common_Dates[i]].ConfirmedCases + 1
    tmp_log_train = tmp_train.apply(np.log)
    # tmp_log_train
    tmp_test = test_set[test_set['Date']==common_Dates[i]].ConfirmedCases + 1
    tmp_log_test = tmp_test.apply(np.log)
    dif = (tmp_log_train.values - tmp_log_test.values)
    squ = dif*dif
    S = S + squ.sum()
    n = n + squ.shape[0]
    # test_set[test_set['Date']==common_Dates[i]]
    # tmp_log_test
RMSLE = np.sqrt((1/n)*S)
print('RMSLE:')
print(RMSLE)

ind = test_dt[test_dt['Country_Region']=='Italy'].ForecastId.index
# ind = test_dt[test_dt['Province_State']=='Hubei'].ForecastId.index
confCases = test_set[ind.min():ind.max()]['ConfirmedCases']
plt.plot(confCases, label = 'Confirmed Cases')
plt.xlabel('Predictions Dates since 19/03')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed Cases')
# ConfirmedCases
fatal = test_set[ind.min():ind.max()]['Fatalities']
plt.figure()
plt.plot(fatal, label = 'Fatalities')
plt.xlabel('Predictions Dates since 19/03')
plt.ylabel('Fatalities')
plt.title('Fatalities')

# Fatalities
# test_dt[test_dt['Country_Region']=='Greece']
# test_dt
submi_dt['ConfirmedCases'] = test_set.ConfirmedCases
submi_dt['Fatalities'] = test_set.Fatalities
submi_dt.to_csv('submission.csv', index = False)