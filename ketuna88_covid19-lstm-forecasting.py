import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
np.random.seed(7)
test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv", parse_dates=["Date"])
train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
train['Province_State'].fillna('None', inplace = True)
confirmed_total= train.groupby(['Date']).agg({'ConfirmedCases' : ['sum']})
fatalities_total = train.groupby(['Date']).aggregate({'Fatalities': ['sum']})
total_date = confirmed_total.join(fatalities_total)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date.plot(ax=ax1)
ax1.set_title("Global confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total.plot(ax=ax2, color='orange')
ax2.set_title("Global fatalities", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)
pivot=pd.pivot_table(train,columns='Country_Region',index='Date',values='ConfirmedCases',aggfunc=np.sum)
pivot_fatality=pd.pivot_table(train,columns='Country_Region',index='Date',values='Fatalities',aggfunc=np.sum)
country_list=[]
value_list=[]
fatality_list=[]
for country in list(pivot.columns):
    country_list.append(country)
    value_list.append(pivot[country].max())
    fatality_list.append(pivot_fatality[country].max())
    new_dict={'Country':country_list,'Confirmed':value_list,'Fatality':fatality_list}
df=pd.DataFrame.from_dict(new_dict)
df.set_index('Country',inplace=True)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
df['Confirmed'].sort_values(ascending=False)[:10].plot(kind='bar',color='blue')
plt.title('Top 10 Countries by Confirmed Cases')
plt.subplot(2,1,2)
df['Fatality'].sort_values(ascending=False)[:10].plot(kind='bar',color='red')
plt.title('Top 10 Countries with Fatalities due to Covid-19')
plt.tight_layout()
top_confirmed=df.sort_values(by='Confirmed',ascending=False)[:10]
list_countries=list(top_confirmed.index)
Confirm_pivot=pd.pivot_table(train,index='Date',columns='Country_Region',
                             values='ConfirmedCases',aggfunc=np.sum)
plt.figure(figsize=(20,16))
colors=['purple','green','blue','yellow','orange','r','m','hotpink','violet','darkgreen','navy','brown']
for i,country in enumerate(list_countries):
    Confirm=Confirm_pivot[Confirm_pivot[country]>0][country].diff().fillna(0)
    Confirm=Confirm[Confirm>0]
    
    plt.subplot(4,3,i+1)
    Confirm.plot(color=colors[i],label=country,markersize=12,lw=5)    
    plt.xticks()
    plt.legend(title='Country')
    plt.title('Number of Daily Cases in {}'.format(country.upper()))
plt.tight_layout()
# generate key for each entitiy to predict for
train['geo_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
train.head()
def get_entity_data_split(geo_id, train_split_factor=1.0):
    data = train[train["geo_id"] == geo_id]
    country = data["Country_Region"].unique()[0]
    province = data["Province_State"].unique()[0]
    
    case = data["ConfirmedCases"].to_numpy()
    fat = data["Fatalities"].to_numpy()
    case = case.reshape((len(case), 1))
    fat = fat.reshape((len(fat), 1))
    
    train_size = int(len(data) * train_split_factor)
    test_size = len(data) - train_size
    
    case_train, case_test = case[0:train_size,:], case[train_size:len(data),:]
    fat_train, fat_test = fat[0:train_size,:], fat[train_size:len(data),:]
    
    return train_size, test_size, case_train, case_test, fat_train, fat_test

def create_dataset_for_lstm(data, look_back=1.0):
    # reshape into X=t and Y=t+1
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    X = np.array(dataX)
    Y = np.array(dataY)
    
    # reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    return X, Y

def calculate_loss(Y, pred):
    return np.sqrt(mean_squared_log_error(Y[0], pred[:,0]))

def train_lstm(train, test=[], look_back=1, epochs=10, batch_size=1, verbose=0):
    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test) if (len(test) > 0) else []
    
    # prep data
    trainX, trainY = create_dataset_for_lstm(train_scaled, look_back=look_back)
    testX, testY = (create_dataset_for_lstm(test_scaled, look_back=look_back)) if (len(test) > 0) else ([], [])
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    # make predictions
    trainPredict = model.predict(trainX).clip(0)
    testPredict = model.predict(testX).clip(0) if (len(test) > 0) else []
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict) if (len(testPredict) > 0) else []
    testY = scaler.inverse_transform([testY]) if (len(testY) > 0) else []
    
    # calculate RMSLE
    trainScore = calculate_loss(trainY, trainPredict)
    testScore = calculate_loss(testY, testPredict) if (len(test) > 0) else 0
    if (verbose):
        print('Train Score: %.2f RMSLE' % (trainScore))
        print('Test Score: %.2f RMSLE' % (testScore))
    
    return {
        "model": model,
        "scaler": scaler,
        "train": train,
        "trainX": trainX,
        "trainY": trainY,
        "trainPredict": trainPredict,
        "trainScore": trainScore,
        "test": test,
        "testX": testX,
        "testY": testY,
        "testPredict": testPredict,
        "testScore": testScore
    }

def plot_lstm(model, scaler, train, test, look_back=1.0, title="Cases"):
    data = np.concatenate((train, test))
    
    # scale
    train = scaler.transform(train)
    test = scaler.transform(test) if (len(test) > 0) else []
    
    # prep data
    trainX, _ = create_dataset_for_lstm(train, look_back=look_back)
    testX, _ = (create_dataset_for_lstm(test, look_back=look_back)) if (len(test) > 0) else ([], [])
    
    # make predictions
    trainPredict = model.predict(trainX).clip(0)
    testPredict = model.predict(testX).clip(0) if (len(testX) > 0) else []
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict) if (len(testPredict) > 0) else []
    
    # plot baseline and predictions
    plt.plot(range(1, len(data) + 1), data, label="Actual")
    plt.plot(range(1 + look_back, len(trainPredict) + look_back + 1), trainPredict, label="Pred Train")
    plt.plot(range(1 + 2*look_back + len(trainPredict) + 1, len(data)), testPredict, label="Pred Test")
    plt.title("Fit of LSTM Preds to Actual " + title)
    plt.ylabel(title)
    plt.xlabel("Days")
    plt.legend()
    plt.show()
# settings
train_split_factor = 0.8
look_back = 1
verbose = 0
epochs = 50

# entities
geo_ids = train["geo_id"].unique()

# train
results_dict = {}
for geo_id in tqdm(geo_ids):
    # get train and test data
    train_size, test_size, case_train, case_test, fat_train, fat_test = get_entity_data_split(geo_id, train_split_factor=train_split_factor)
    
    if (verbose): print("train case")
    case_results = train_lstm(case_train, test=case_test, look_back=look_back, epochs=epochs, verbose=verbose)
    if (verbose): print("train fat")
    fat_results = train_lstm(fat_train, test=fat_test, look_back=look_back, epochs=epochs, verbose=verbose)
    
    results_dict[geo_id] = (case_results, fat_results)
def plot_result(geo_id, results_dict=results_dict, features=("case", "fat")):
    # get train and test data
    train_size, test_size, case_train, case_test, fat_train, fat_test = get_entity_data_split(geo_id, train_split_factor=train_split_factor)

    # get results
    case_results, fat_results = results_dict[geo_id]

    # print and plot
    print(geo_id)
    # case
    if ("case" in features):
        print("Confirmed Cases:")
        print("  trainScore: ", case_results["trainScore"])
        print("  testScore: ", case_results["testScore"])
        plot_lstm(case_results["model"], case_results["scaler"], case_results["train"], case_results["test"], look_back=look_back, title="Confirmed Cases")
    # fat
    if ("fat" in features):
        print("Fatalities:")
        print("  trainScore: ", fat_results["trainScore"])
        print("  testScore: ", fat_results["testScore"])
        plot_lstm(fat_results["model"], fat_results["scaler"], fat_results["train"], fat_results["test"], look_back=look_back, title="Fatalities")
results_list = list(results_dict.items())
len(results_list)
# sort results by case_model testScore asc
results_list.sort(key=lambda entity: entity[1][0]["testScore"])
for i in range(0, 3): plot_result(results_list[i][0], features=("case"))
# sort results by fat_model testScore asc
results_list.sort(key=lambda entity: entity[1][1]["testScore"])
try:
    for i in range(0, 3): plot_result(results_list[i][0], features=("fat"))
except:
    print("Result")
allTrainY = []
allTrainPredict = []

allTestY = []
allTestPredict = []

# cumulate Y_true and Y_pred for train and test
for geo_id, results in results_dict.items():
    
    for result in results: # case and fat
        # train
        trainY = result["trainY"]
        trainPredict = result["trainPredict"]
        if (trainY.size == trainPredict.size):
            allTrainY.extend(trainY)
            allTrainPredict.extend(trainPredict)
        else:
            print("Warning: trainY.size != trainPredict.size: ", trainY.size, "!=", trainPredict.size)

        # test
        testY = result["testY"]
        testPredict = result["testPredict"]
        if (testY.size == testPredict.size):
            allTestY.extend(testY)
            allTestPredict.extend(testPredict)
        else:
            print("Warning: testY.size != testPredict.size: ", testY.size, "!=", testPredict.size)

# make nparrays
allTrainY = np.array(allTrainY)
allTrainPredict = np.array(allTrainPredict)
allTestY = np.array(allTestY)
allTestPredict = np.array(allTestPredict)
# reshape for calculate_loss()
allTrainY = allTrainY.reshape((1, allTrainY.size))
allTrainPredict = allTrainPredict.reshape((allTrainPredict.size, 1))
allTestY = allTestY.reshape((1, allTestY.size))
allTestPredict = allTestPredict.reshape((allTestPredict.size, 1))

# calculate overall score (loss) for train and test
overallTrainScore = calculate_loss(allTrainY, allTrainPredict)
overallTestScore = calculate_loss(allTestY, allTestPredict)
print("overallTrainScore: ", overallTrainScore)
print("overallTestScore: ", overallTestScore)
