import numpy as np 
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from haversine import haversine
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed

#set random seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)
warnings.filterwarnings("ignore")
#load train and test data
train =  pd.read_csv('../input/train.csv', nrows = 6_000_000)
test =  pd.read_csv('../input/test.csv')
train.head()
#no of null values in train data. There are null vlaue sin dropoff attributes.
train.isnull().sum()
#no of null values in test data. There are no null values which was expected.
test.isnull().sum()
#the train data
train.describe()
#datatypes of the train data
train.dtypes
#since there are lots training data, the no. of null value records is negligible. So we are dropping
#the null value records.
train.dropna(subset=['dropoff_latitude','dropoff_longitude'], inplace=True)
train.isnull().sum()
#with little bit of googling, we can find the exact lattitude and longitude values for NYC. So we can 
#filter out only those records which are within these bounds. We filter both pickup and 
#dropoff atributes.
def clean_lats_long(full_data):    
    full_data = full_data[(-76 <= full_data['pickup_longitude']) & (full_data['pickup_longitude'] <= -72)]
    full_data = full_data[(-76 <= full_data['dropoff_longitude']) & (full_data['dropoff_longitude'] <= -72)]
    full_data = full_data[(38 <= full_data['pickup_latitude']) & (full_data['pickup_latitude'] <= 42)]
    full_data = full_data[(38 <= full_data['dropoff_latitude']) & (full_data['dropoff_latitude'] <= 42)]
    return full_data

train= clean_lats_long(train)
print(train.shape)
#test = clean_lats_long(test)
#the passenger count has s´few records with count more than 10 and less than 1. Usually even a SUV kind
#of taxi woud take max of 10 people. Definitley counts that are less than 1 are wrong. So we drop
#these records
print(train.passenger_count[train.passenger_count > 10].count())
print(train.passenger_count[train.passenger_count < 10].count())
print(train.passenger_count[train.passenger_count == 0].count())
train = train[(train.passenger_count < 10)&(train.passenger_count > 0)]
print(train.shape)
#usually taxis start off with certain amount of fixed cost. We may have to pay some money just to get into
#a taxi. So there can be no trips tha cost less than 1$. So we drop them as well.
print(train.fare_amount[train.fare_amount < 1].count())
train = train[train.fare_amount >= 1]
print(train.shape)
#haversine distance is the distance between two set of lattitude and longitude points. Taxi fare is
#directly influenced by the time travelled and the distance. SO distance as a feature is a good addition.
def drop_latlong_column(df):
    df.drop(columns=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude'
                     ,'dropoff_longitude'],inplace=True)
    return df

def haversine_lat_long(df):
    df['dist_travel']=df.apply(lambda x:haversine((x['pickup_latitude'],
                                                                 x['pickup_longitude']),
                                                                 (x['dropoff_latitude'],
                                                                 x['dropoff_longitude']),
                                                   unit='mi'),axis=1)
    return drop_latlong_column(df)

train = haversine_lat_long(train)
test = haversine_lat_long(test)
train.head()
#once dist is calculated, we can now plot the fare amount and dist and make some exploratory analysis.
#we just use he last 10000 records to save memory and time.
train[:10000].plot.scatter(x="fare_amount", y="dist_travel")
print(train.fare_amount.corr(train.dist_travel))
#form the above plot, it is clear that some fare amounts and dist travelled dont add up. THere are
#some points which have high fare with very little or no dist travelled and vice versa. But the plot 
#is only for 10000 records, so in general we could say fare amount less than 10$ with dist travelled
#more than 50 as well fare amount more than 150$ and dis less than 5 can be dropped.
train.drop(train[(train.fare_amount < 10) & (train.dist_travel>50)].index, inplace=True)
train.drop(train[(train.fare_amount >150) & (train.dist_travel<5)].index, inplace=True)
train.shape
#to process data and time feature. We convert it to type datatime first and extract all the values 
#separately like, hour, day, etc. This helpful tonascertain when the ride was taken. For example, during
#peak hours the fare can be higher than usual. Secondly, we convert hours to bins(evening, morning,afternoon,
#late night). THis again determines adn tells us when the ride as taken. ALso makes sense to group similar
#hours to a certain bin so that they are treted similarly. 
def process_datetime(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    df['year'] = df.pickup_datetime.dt.year
    df['day'] = df.pickup_datetime.dt.day
    df['hour'] = df.pickup_datetime.dt.hour
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour_bin'] = pd.cut(df.hour,bins=4, labels=['LN','MO','AN','EV'])
    df.drop(columns=['pickup_datetime','hour'],inplace=True)
    return df

train = process_datetime(train)
test = process_datetime(test)

train.head()
#drop the columns key in both datasets
train.drop(columns=['key'],inplace=True)
test.drop(columns=['key'],inplace=True)
train.head()
#create dummies for hour_bin feature
train= pd.get_dummies(train, columns=['hour_bin'],drop_first=True)
test= pd.get_dummies(test, columns=['hour_bin'],drop_first=True)
train.head()
#separate the target variable and drop it from train set
target = train.fare_amount
train.drop(columns=['fare_amount'],inplace=True)
train.head()
#convert to np array and do scaling. We use standard scaling so that data is centered and also has a 
#standard deviation of 1. scaling is important for gradient based learning algorithms. 
train = np.array(train)
test = np.array(test)
target = np.array(target)
scaler = StandardScaler(copy=False)
scaler.fit(train)
scaler.transform(train)
scaler.transform(test)
#split to train and validation sets.
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3)
print(len(x_train))
#Train the model using keras deep learning.
early_stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-3, patience=3)
callback=[early_stop]
adam = Adam(lr=0.0001)

model = Sequential() 
model.add(Dense(100, activation='relu', input_shape=(train.shape[1],)))
model.add(Dropout(0.6))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=adam, metrics=['mae'])
history = model.fit(x_train,y_train,batch_size=256, epochs=25, verbose=1, callbacks=callback,
         validation_data=(x_val, y_val), shuffle=True)
import matplotlib.pyplot as plt

# summarize history for loss using learning curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#predict the rmse for validation set. 
from sklearn.metrics import mean_squared_error
y_pred = np.array(model.predict(x_val))
print(sqrt(mean_squared_error(y_val, y_pred)))
#submit the submission results. The results could be improved a lot with hyperparameter tunig using
#gridsearch or using an ensmble model with final result based on outputs of different regressors.
#Other improvements like using important spots in NYC like airports,etc can be added to features.
sub =  pd.read_csv('../input/sample_submission.csv')
y_sub = model.predict(test)
sub.fare_amount = y_sub
sub.to_csv('Submission1.csv',index=False)