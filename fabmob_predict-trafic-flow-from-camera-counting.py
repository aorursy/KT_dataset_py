import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pickle
import math
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
# Global setting
plt.style.use('seaborn')
file = "/kaggle/input/traffic-counting-using-cameras/telraam.csv"
# Describe Data
def describeData(data):
    print("shape = {}".format(data.shape))
    description = data.describe().T
    description["isNull"] = data.isnull().sum()
    print(description)
# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return math.sqrt(mean_squared_error(actual, predicted))
# Load the data
##########################################
df = pd.read_csv(file, parse_dates = False, header = None,sep=',')
df.columns=['time','id','timezone','pct_up','pedestrian','bike','car','lorry','pedestrian_lft','bike_lft','car_lft','lorry_lft','pedestrian_rgt','bike_rgt','car_rgt','lorry_rgt','car_speed_00','car_speed_10','car_speed_20','car_speed_30','car_speed_40','car_speed_50','car_speed_60','car_speed_70']
df['time'] = pd.to_datetime(df['time'],errors='coerce',utc=True)  
df['time'] = df['time'].fillna(method = 'ffill')
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
#describeData(df)
# Skip the first row
########################################### 
minDate = '2020-06-16 00:00:00+00'
df = df.loc[df['time']>minDate]

# Reindex the data
########################################### 
df.set_index(['time'],drop=False,inplace=True)
all_freq = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
df = df.reindex(all_freq)
df['timezone'].fillna('Europe/Paris',inplace=True)
df['time'] = df.index
df['localtime']=df.time.map(lambda t: t.tz_convert(tz = 'Europe/Paris'))
df['dayofweek'] = df['localtime'].dt.dayofweek
df['day'] = df['localtime'].dt.day
df['hour'] = df['localtime'].dt.hour
# Fill missing values
###########################################
for col in ['car','pedestrian','lorry','bike','pedestrian_lft','bike_lft','car_lft','lorry_lft','pedestrian_rgt','bike_rgt','car_rgt','lorry_rgt']:
    df['adjust_'+col] = df.groupby(['dayofweek','hour'])[col].transform(lambda x: x.fillna(x.median()))
    df[col].fillna(0,inplace=True)
df.fillna(0,inplace=True)

# See adjusted data versus row data
###########################################
df['car'].plot(title='Number of car')
plt.show()
df['adjust_car'].plot(title='Adjusted number of car ')
plt.show()
# Estimate the average speed
###########################################
df['speed']=df['car_speed_00']*5+df['car_speed_10']*15+df['car_speed_20']*25+df['car_speed_30']*35+df['car_speed_40']*45+df['car_speed_50']*55+df['car_speed_60']*65+df['car_speed_70']*75
df['speed'] = df['speed'] / (df['car_speed_00']+df['car_speed_10']+df['car_speed_20']+df['car_speed_30']+df['car_speed_40']+df['car_speed_50']+df['car_speed_60']+df['car_speed_70'])
df['speed'].plot(title='Speed')
plt.show()
left = pd.pivot_table(df, index="hour", values=['car_lft'],columns=['dayofweek'], aggfunc=np.sum,margins=False,dropna=False)
right = pd.pivot_table(df, index="hour", values=['car_rgt'],columns=['dayofweek'], aggfunc=np.sum,margins=False,dropna=False)
left.columns = ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']
right.columns = ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']

barWidth = 0.45

for day in ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']:
    r1 = np.arange(len(left[day]))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, left[day], color='#d35400', width=barWidth, edgecolor='white', label='car count left')
    plt.bar(r2, right['Monday'], color='#2e86c1', width=barWidth, edgecolor='white', label='car count right')
    plt.xlabel('Hour', fontweight='bold')
    plt.legend()
    plt.title(day)
    plt.show()
# max / median average speed by hour
###########################################
result = pd.pivot_table(df, index="hour", values=['speed'],columns=['dayofweek'], aggfunc=np.max,margins=False,dropna=False)
result.columns = ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']
result.plot(title='{} : max average speed by hour'.format('Speed'))
plt.show()

result = pd.pivot_table(df, index="hour", values=['speed'],columns=['dayofweek'], aggfunc=np.median,margins=False,dropna=False)
result.columns = ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']
result.plot(title='{} : median average speed by hour'.format('Speed'))
plt.show()
# Mean number by hour
###########################################
for col in ['car','pedestrian','lorry','bike']:
    result = pd.pivot_table(df, index="hour", values=[col],columns=['dayofweek'], aggfunc=np.mean,margins=False,dropna=False)
    result.columns = ['Monday','Tuesday','Wenesday','Thursday','Friday','Saturday','Sunday']
    result.plot(title='{} : mean number by hour'.format(col))
    plt.show()
df['car'].plot(title='Number of car')
plt.show()
# Decomposition
###########################################
 
d=sm.tsa.seasonal_decompose(df.adjust_car,period=24)
figure = d.plot()
plt.show()
# Select a window of time with clean data 
###########################################

minDate = '2020-06-16 10:00:00+00'
maxDate = '2020-07-21 00:00:00+00'

# Select a window of time with clean data 
# AND
# Split train and test
###########################################
train = df.adjust_car.loc[df.index<'2020-07-19 00:00:00+00'].copy()
test = df.adjust_car.loc[(df.index>='2020-07-19 00:00:00+00')&(df.index<'2020-07-21 00:00:00+00')].copy().iloc[:48]
train_and_test = pd.concat([train,test])
print(train.shape)
print(test.shape)
# SARIMA
###########################################
model = sm.tsa.statespace.SARIMAX(train, trend='n', order=(1,0,0), seasonal_order=(2,1,0,24))
model_sarima = model.fit()
print(model_sarima.summary())
forecast_sarima = model_sarima.predict(start = test.index[0], end= test.index[-1], dynamic= True) 
plt.plot(test,label="true")
plt.plot(forecast_sarima,label="forecast")
plt.legend(loc='upper left')
plt.title("SARIMA - Forecast of number of car")
plt.show()

# root mean squared error or rmse
print(measure_rmse(test, forecast_sarima))
# ExponentialSmoothing
###########################################

from statsmodels.tsa.api import ExponentialSmoothing
model_exp = ExponentialSmoothing(train.values, trend='add', seasonal='add', seasonal_periods=24).fit()
forecast_exp = model_exp.forecast(len(test))
forecast_exp = pd.DataFrame(data=forecast_exp,index=test.index)
plt.plot(test,label="true")
plt.plot(test.index,forecast_exp,label="forecast")
plt.legend(loc='upper left')
plt.title("ExponentialSmoothing - Forecast of number of car")
plt.show()

# root mean squared error or rmse
print(measure_rmse(test, forecast_exp))
# LSTM
###########################################

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


# Window size = number of previous values to predict the next value
WINDOW_SIZE = 10

train_serie = train.values.reshape(-1, 1)
all_serie = train_and_test.values.reshape(-1, 1)

#  MinMaxScaler
scaler = MinMaxScaler()
scaled_all_serie = scaler.fit_transform(all_serie)
scaled_train_serie = scaler.transform(train_serie)                     
#scaled_close = scaled_close[~np.isnan(scaled_close)]
#scaled_close = scaled_close.reshape(-1, 1)
print("Train shape = {}".format(scaled_train_serie.shape))
print("All shape = {}".format(scaled_all_serie.shape))
print("nan values ? {}".format(np.isnan(scaled_train_serie).any()))

def generateSequence(sequence,backward):
    X, y = list(), list()
    for i in range(sequence.shape[0]-backward):
        seq_x, seq_y = sequence[i:i+backward], sequence[i+backward]
        X.append(seq_x)
        y.append(seq_y)
    X=np.array(X)
    y=np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X,y
    
X,y = generateSequence(scaled_train_serie,WINDOW_SIZE)
print("X shape = {}".format(X.shape))
print("y shape = {}".format(y.shape))


model = Sequential()
model.add(LSTM(12, activation='relu', input_shape=(WINDOW_SIZE, 1)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


#  Compile
model.compile(
    loss='mse', 
    optimizer='adam'
)

BATCH_SIZE = 64

#  Compile
history = model.fit(
    X, 
    y, 
    epochs=50, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


X,y = generateSequence(scaled_all_serie,WINDOW_SIZE)
y_predicted = model.predict(X)
y_inverse = scaler.inverse_transform(y)
y_predicted_inverse = scaler.inverse_transform(y_predicted)

forecast_lstm = y_predicted_inverse[-48:].ravel()
plt.plot(test,label="true")
plt.plot(test.index,forecast_lstm,label="forecast")
plt.legend(loc='upper left')
plt.title("LSTM - Forecast of number of car")
plt.show()

# root mean squared error or rmse
print(measure_rmse(test, forecast_lstm))
## Comparison
plt.plot(test,label="true",linestyle='dashed')
plt.plot(test.index,forecast_lstm,label="forecast_lstm")
plt.plot(test.index,forecast_exp,label="forecast_exp")
plt.plot(test.index,forecast_sarima,label="forecast_sarima")
plt.legend(loc='upper left')
plt.title("Forecast of number of car")
plt.show()

# root mean squared error or rmse
print("root mean squared error of SARIMA model     = {}".format(measure_rmse(test, forecast_sarima)))
print("root mean squared error of EXP SMOOTH model = {}".format(measure_rmse(test, forecast_exp)))
print("root mean squared error of LSTM model       = {}".format(measure_rmse(test, forecast_lstm)))