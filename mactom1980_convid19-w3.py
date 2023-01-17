# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

files = [None]*3
i = 0
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

print('train inst., test inst')
print(train_dt.shape[0], test_dt.shape[0])
from pandas.api.types import CategoricalDtype

# append train test sets
test_train_dt = train_dt.append(test_dt, sort=False)

# make Date Categorical (ordered)
uniqueDates = list(test_train_dt['Date'].unique())
nrOfTrain_Test_dates = len(uniqueDates)

# dates strating from 0
cat_type_date = CategoricalDtype(categories = uniqueDates , ordered=True)
test_train_dt.Date = test_train_dt.Date.astype(cat_type_date).cat.codes.astype(float)
# test_train_dt[test_train_dt.Country_Region=='Denmark']
test_train_dt.Province_State = test_train_dt.Province_State.astype("category").cat.codes.astype(float)
# Country_Region categorical
test_train_dt.Country_Region = test_train_dt.Country_Region.astype("category").cat.codes.astype(float)
# country starts with 0
max_country = test_train_dt.Country_Region.unique().max()
max_State = test_train_dt.Province_State.unique().max()
print('max_country, max_state')
print(max_country,max_State)

# make COuntryIndx unique
test_train_dt['CountrIndx']=0

# newCases_conf, neewCases_fat, mult_conf, mult_fat

test_train_dt['newCases_conf'] = 0.0
test_train_dt['newCases_fat'] = 0.0
# test_train_dt['mult_conf']
# test_train_dt['mult_fat']
nrOfTrainTestInst = test_train_dt.shape[0]
# for i in range(nrOfTrainTestInst):

CountryIndx_Array = np.zeros((nrOfTrainTestInst,1))

# give to the country that have not states the country id
CountryIndx_Array[test_train_dt['Province_State']==-1,0] = test_train_dt[test_train_dt['Province_State']==-1].Country_Region.values
CountryIndx_Array[test_train_dt['Province_State']!=-1,0] = test_train_dt[test_train_dt['Province_State']!=-1].Province_State.values
CountryIndx_Array[test_train_dt['Province_State']!=-1,0] = CountryIndx_Array[test_train_dt['Province_State']!=-1,0] + (max_country + 1)
test_train_dt.CountrIndx = CountryIndx_Array
train_set = test_train_dt[:end_train]
test_set = test_train_dt[end_train:]
nrOfTrainSamples = train_set.shape[0]
# train_set['ConfirmedCases'][0:10]
train_set['DaySinceOutbr']=0
max_tr_date = train_set['Date'].unique().max()
nrTrainInst = train_set.shape[0]
sinceTheOutbr_array = np.zeros((nrTrainInst,1))
for i in range(nrTrainInst):
    if train_set['Date'][i] == 0:
        count = 0
    if train_set['ConfirmedCases'][i] > 0 :
        count = count + 1
#         train_set['DaySinceOutbr'][i] = count
        sinceTheOutbr_array[i] = count
    if train_set['Date'][i] > 0:
        tmp_11 = train_set['ConfirmedCases'][i] - train_set['ConfirmedCases'][i-1]
        if tmp_11 < 0:
            train_set.set_value(i,'ConfirmedCases', train_set['ConfirmedCases'][i-1])
        tmp_11 = train_set['ConfirmedCases'][i] - train_set['ConfirmedCases'][i-1]
        train_set.set_value(i,'newCases_conf', tmp_11)
        
        tmp_12 = train_set['Fatalities'][i] - train_set['Fatalities'][i-1]
        if tmp_12 < 0:
            train_set.set_value(i,'Fatalities', train_set['Fatalities'][i-1])
        tmp_12 = train_set['Fatalities'][i] - train_set['Fatalities'][i-1]
        train_set.set_value(i,'newCases_fat', tmp_12)
#         tmp_12 = train_set['Fatalities'][i] - train_set['Fatalities'][i-1]
#         train_set.set_value(i,'newCases_fat', tmp_12)
        
train_set['DaySinceOutbr'] = sinceTheOutbr_array
test_set['DaySinceOutbr']=0
nrTestInst = test_set.shape[0]
lastTrainDate = train_set['Date'].unique().max()

# for i in range(nrTestInst):
    
train_set[train_set['newCases_conf']<0]
train_set[9280:9295]
first_test_day = test_set['Date'].unique().min()
nrOfTestInst = test_set.shape[0]
sinceTheOutTest_arr = np.zeros((nrOfTestInst,1))

for i in range(nrOfTestInst):
    if first_test_day>lastTrainDate:
        print('ton ipiame')
        if test_set['Date'][i] == first_test_day:
            
            # find the since the outbraek from train set
            country_reg = test_set['CountrIndx'][i]
            tmp1 = train_set[train_set['CountrIndx'] == country_reg]
            # lastTrainDate
            tmpSinceTheOut = tmp1[tmp1.Date==lastTrainDate].DaySinceOutbr.values
            count = tmpSinceTheOut + 1
            sinceTheOutTest_arr[i] = count
        else:
            count = count + 1
            sinceTheOutTest_arr[i] = count
    else:
        if test_set['Date'][i] <= lastTrainDate:
            courentDate = test_set['Date'][i]
            #find the since the outbreak from train
            country_reg = test_set['CountrIndx'][i]
            tmp1 = train_set[train_set['CountrIndx'] == country_reg]
            # lastTrainDate
            tmpSinceTheOut = tmp1[tmp1.Date==courentDate].DaySinceOutbr.values
            sinceTheOutTest_arr[i] = tmpSinceTheOut
            count = tmpSinceTheOut
        else:
            count = count + 1
            sinceTheOutTest_arr[i] = count
            
test_set['DaySinceOutbr'] = sinceTheOutTest_arr
time_span = 7;
#confirmedCases = time_span, fatalities = time_span
#DaySinceTheOutbreak = time_span, CountryIndx = 1, CourentDay = time_span
#newCases_conf= time_span, newCases_fat= time_span
# nuberOfFeatures = 6*time_span  + 1
nuberOfFeatures = 7*time_span
# train_set.shape[0]
# nrOfTrainDates = train_set.Date.unique().shape[0]

nrOfTrainData = train_set.shape[0]

# find the unique countries/staes

countrUniques = train_set.CountrIndx.unique()

# nrOfTrainData
X = np.zeros((nrOfTrainData,nuberOfFeatures))
Y = np.zeros((nrOfTrainData,2))

# calcuate the labels for the train set

inst_count = 0;


for i in countrUniques:

    country = train_set[train_set['CountrIndx']==i]
    nrOfdaysPerCountry = country.shape[0]
    
    for j in range(nrOfdaysPerCountry-time_span):
        
        X[inst_count:inst_count+1, 0:time_span] = country.ConfirmedCases[j:time_span+j].values
        X[inst_count:inst_count+1, time_span:2*time_span] = country.Fatalities[j:time_span+j].values
        
        X[inst_count:inst_count+1, 2*time_span:3*time_span] = country.DaySinceOutbr[j:time_span+j].values
        
        X[inst_count:inst_count+1,3*time_span:4*time_span] = country.Date[j:time_span+j].values
        
        X[inst_count:inst_count+1,4*time_span:5*time_span] = country.newCases_conf[j:time_span+j].values
        X[inst_count:inst_count+1,5*time_span:6*time_span] = country.newCases_fat[j:time_span+j].values
        
#         X[inst_count:inst_count+1,6*time_span:7*time_span] = country.CountrIndx[j:j+1].values
        X[inst_count:inst_count+1,6*time_span:7*time_span] = country.CountrIndx[j:time_span+j].values
        
#         X[inst_count:inst_count+1, 2*time_span+1:2*time_span+2] = country.DaySinceOutbr[j:j+1].values
        
        Y[inst_count:inst_count+1,0:1] = country.ConfirmedCases[time_span+j:time_span+j+1].values
        Y[inst_count:inst_count+1,1:2] = country.Fatalities[time_span+j:time_span+j+1].values
        
        inst_count = inst_count + 1

X[250,:]
def log_10(x):
#     return 1 / (1 + np.exp(x))
    return np.log10(x+1)
def min_max_scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)
X_scale = np.zeros((nrOfTrainData,nuberOfFeatures))
Y_scale = np.zeros((nrOfTrainData,2))
X_scale = X
X_scale[:,0:2*time_span] = log_10(X[:,0:2*time_span])
X_scale[:,2*time_span:4*time_span] = min_max_scale(X[:,2*time_span:4*time_span], 0, test_set['Date'].unique().max())

X_scale[:,4*time_span:6*time_span] = log_10(X[:,4*time_span:6*time_span])

X_scale[:,6*time_span:7*time_span] = min_max_scale(X[:,6*time_span:7*time_span], 0, train_set['CountrIndx'].unique().max())
# X_scale[:,3*time_span+1:3*time_span+2] = min_max_scale(X[:,3*time_span:3*time_span+1], 0, test_set['Date'].unique().max())

Y_scale = log_10(Y)

# X[200:200+1,6*time_span:7*time_span] = country.CountrIndx[j:time_span+j].values
country.CountrIndx[j:time_span+j].values
X[150,:].shape
# from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.optimizers import rmsprop, Adam
import numpy as np
import numpy.random as nr
# from tensorflow import set_random_seed
import matplotlib.pyplot as plt

from keras.layers import Dropout, LeakyReLU
nrOfTrainSamples
# countrUniques.shape[0]
nn = models.Sequential()
nn.add(layers.Dense(6*time_span, activation = 'softplus', 
                    input_shape = (nuberOfFeatures, ), 
                   kernel_regularizer=regularizers.l2(0.001)))

# nn.add(layers.Dense(6*time_span, activation = 'softplus',  
#                    kernel_regularizer=regularizers.l2(0.001)))

nn.add(layers.Dense(5*time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))

nn.add(layers.Dense(4*time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(3*time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(2*time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(time_span,activation = 'softplus', 
                   kernel_regularizer=regularizers.l2(0.001)))
# nn.add(Dropout(rate = 0.2))
# nn.add(layers.Dense(countrUniques.shape[0], activation = 'relu',
#                    kernel_regularizer=regularizers.l2(0.01)))
# nn.add(layers.Dense(countrUniques.shape[0], activation = 'softplus',
#                    kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(2*time_span, activation = 'softplus',  
                   kernel_regularizer=regularizers.l2(0.001)))
# nn.add(Dropout(rate = 0.2))
nn.add(layers.Dense(3*time_span, activation = 'softplus', 
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(4*time_span, activation = 'softplus', 
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(5*time_span, activation = 'softplus', 
                   kernel_regularizer=regularizers.l2(0.001)))
nn.add(layers.Dense(6*time_span, activation = 'softplus', 
                   kernel_regularizer=regularizers.l2(0.001)))
# nn.add(layers.Dense(7*time_span, activation = 'softplus', 
#                    kernel_regularizer=regularizers.l2(0.001)))
# nn.add(Dropout(rate = 0.2))

nn.add(layers.Dense(2))
nn.summary()
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
nn.compile(optimizer = 'Adam', loss = 'mean_squared_error', 
                metrics = ['mae'])
history = nn.fit(X_scale, Y_scale,
                  epochs = 800, batch_size = nrOfdaysPerCountry, verbose = 0, shuffle = False) #,validation_data = (X, Y))
# history = nn.fit(X_scale, Y,
#                   epochs = 600, batch_size = nrOfdaysPerCountry, verbose = 0) #,validation_data = (X, Y))
history.history['loss'][-1]
plot_loss(history)
nrOfTestInst = test_set.shape[0]

# X_test = np.zeros((nrOfTestInst, nuberOfFeatures))
X_test_scale = np.zeros((1,nuberOfFeatures))
# X_test_scale = np.zeros((nrOfTestInst, nuberOfFeatures))

# the train dates
testDates = test_set.Date.unique()
firstTestDay = testDates.min()

test_set['Date'].unique().min()

for i in range(nrOfTestInst): #nrOfTestInst
#     i=1
    country_index = test_set.CountrIndx[i:i+1].values[0]

    test_instance = test_set[i:i+1]
    test_date = test_instance.Date.values[0]

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

    tmp_count = train_set[train_set['CountrIndx']==country_index]

    tmp_count1 = tmp_count[tmp_count['Date']>=start_previous_days]
    tmp_train_set_days = tmp_count1[tmp_count1['Date']<=start_previous_days+train_set_days-1]

    test_set_start_day = test_date - test_set_days
    test_set_end_day = test_date - 1

    tmp_test = test_set[test_set.CountrIndx==country_index]

    tmp_test1 = tmp_test[tmp_test.Date>=test_set_start_day]
    tmp_test_set_days = tmp_test1[tmp_test1.Date<=test_set_end_day]

    features_df = tmp_train_set_days.append(tmp_test_set_days)

    #     X_test[i:i+1, 0:time_span] = features_df.ConfirmedCases[:].values
    #     X_test[i:i+1, time_span:2*time_span] = features_df.Fatalities[:].values

    #     X_test[i:i+1, 2*time_span:3*time_span] = features_df.DaySinceOutbr[:].values

    #     X_test[i:i+1, 3*time_span:3*time_span+1] = features_df.CountrIndx[0:1].values

    X_test_scale[0, 0:time_span] = log_10(features_df.ConfirmedCases[:].values)
    X_test_scale[0, time_span:2*time_span] = log_10(features_df.Fatalities[:].values)

    X_test_scale[0, 2*time_span:3*time_span] = min_max_scale(features_df.DaySinceOutbr[:].values,0, test_set['Date'].unique().max())
    X_test_scale[0, 3*time_span:4*time_span] = min_max_scale(features_df.Date[:].values, 0, test_set['Date'].unique().max())

    X_test_scale[0, 4*time_span:5*time_span] = log_10(features_df.newCases_conf[:].values)
    X_test_scale[0, 5*time_span:6*time_span] = log_10(features_df.newCases_fat[:].values)

    X_test_scale[0, 6*time_span:7*time_span] = min_max_scale(features_df.CountrIndx[:].values, 0, train_set['CountrIndx'].unique().max())

    #     X_scale[:,2*time_span:3*time_span] = min_max_scale(X[:,2*time_span:3*time_span], 0, test_set['Date'].unique().max())
    #     X_scale[:,3*time_span:3*time_span+1] = min_max_scale(X[:,3*time_span:3*time_span+1], 0, train_set['CountrIndx'].unique().max())
    #     X_test_scale[i,:] = sigmoid(X_test[i,:])

    #     prediction = nn.predict(X_test[i:i+1])
    prediction = nn.predict(X_test_scale)
    prediction = 10**prediction -1
    #     print(prediction)
    #     if prediction[0,0] < 0:
    #         prediction[0,0] = 0
    #     if prediction[0,1] < 0:
    #         prediction[0,1] = 0
    test_set.set_value(i, 'ConfirmedCases', round(prediction[0,0]))
    test_set.set_value(i, 'Fatalities', round(prediction[0,1]))

    # compute newConfirmed and newFat

    if test_set['Date'][i] == firstTestDay:
        # the first day of each country
        # find the country
        tmpCountry_ind = test_set['CountrIndx'][i]
        tmp001 = train_set[train_set['CountrIndx']==tmpCountry_ind]
        trainConf = tmp001[tmp001.Date==firstTestDay].ConfirmedCases.values[0]
        trainFat = tmp001[tmp001.Date==firstTestDay].Fatalities.values[0]

        # if newCases is negative give the previous value
        newCASES = round(prediction[0,0]) - trainConf
        newFAT = round(prediction[0,1]) - trainFat
        if newCASES<0:
#             test_set.set_value(i, 'ConfirmedCases', trainConf+1)
            test_set.set_value(i, 'newCases_conf', 0)
        else:
            test_set.set_value(i, 'newCases_conf', newCASES)
        if newFAT<0:
#             test_set.set_value(i, 'Fatalities', trainFat)
            test_set.set_value(i, 'newCases_fat', 0)
        else:
            test_set.set_value(i, 'newCases_fat', newFAT)
    else:
        newCASES = round(prediction[0,0]) - test_set['ConfirmedCases'][i-1]
        newFAT = round(prediction[0,1]) - test_set['Fatalities'][i-1]
        if newCASES<0:
#             test_set.set_value(i, 'ConfirmedCases', test_set['ConfirmedCases'][i-1])
            test_set.set_value(i, 'newCases_conf', 0)
        else:
            test_set.set_value(i, 'newCases_conf', newCASES)
        if newFAT<0:
#             test_set.set_value(i, 'Fatalities', test_set['Fatalities'][i-1])
            test_set.set_value(i, 'newCases_fat', 0)
        else:
            test_set.set_value(i, 'newCases_fat', newFAT)
#end for

# find the overlap between test and train
country_index = 0
trainDates = train_set[train_set['Country_Region']==0].Date
testDates = test_set[test_set['Country_Region']==0].Date
common_Dates = np.intersect1d(trainDates, testDates)

nrOfCommon_Dates = common_Dates.shape[0]

S = 0
n = 0

for i in range(nrOfCommon_Dates):
    
    tmp_train = train_set[train_set['Date']==common_Dates[i]].ConfirmedCases + 1
    tmp_log_train = tmp_train.apply(np.log)
    
    tmp_test = test_set[test_set['Date']==common_Dates[i]].ConfirmedCases + 1
    tmp_log_test = tmp_test.apply(np.log)
    dif = (tmp_log_train.values - tmp_log_test.values)
    squ = dif*dif
    
    tmp_train2 = train_set[train_set['Date']==common_Dates[i]].Fatalities + 1
    tmp_log_train2 = tmp_train2.apply(np.log)
    
    tmp_test2 = test_set[test_set['Date']==common_Dates[i]].Fatalities + 1
    tmp_log_test2 = tmp_test2.apply(np.log)
    dif2 = (tmp_log_train2.values - tmp_log_test2.values)
    squ2 = dif2*dif2
    
    S = S + squ.sum() + squ2.sum()
    n = n + squ.shape[0] + squ2.shape[0]
    
RMSLE = np.sqrt((1/n)*S)
print('RMSLE:')
print(RMSLE)
X_test_scale[0,:]
test_set[test_set['ConfirmedCases']!=7]
test_set[90:110]
ind = test_dt[test_dt['Country_Region']=='France'].ForecastId.index
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
submi_dt['ConfirmedCases'] = test_set.ConfirmedCases
submi_dt['Fatalities'] = test_set.Fatalities
submi_dt.to_csv('submission.csv', index = False)