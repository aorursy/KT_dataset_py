import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.keras import models, layers, utils, optimizers, callbacks

from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from numpy.random import seed

seed(1)

tf.random.set_seed(1)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ndls_Temp_org_full=pd.read_csv("/kaggle/input/delhi-weather-data/testset.csv")

ndls_Temp_org_full
print(ndls_Temp_org_full.dtypes)

print("================================================")

print(ndls_Temp_org_full.info())
ndls_Temp_org_full.describe()
print("Before modification")

print(ndls_Temp_org_full.columns)

ndls_Temp_org_full.columns=ndls_Temp_org_full.columns.str.replace("_","")

ndls_Temp_org_full.columns=ndls_Temp_org_full.columns.str.replace(" ","")

ndls_Temp_org_full.columns

print("=================================")

print("After modification")

print(ndls_Temp_org_full.columns)
print(ndls_Temp_org_full["conds"].value_counts())

print("No of unique conditions:",len(ndls_Temp_org_full["conds"].unique()))
ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Widespread Dust", "Blowing Sand", 

                                                           "Sandstorm", "Volcanic Ash" ,

                                                            "Light Sandstorm"], "Dust")

ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Fog", "Shallow Fog", "Partial Fog",

                                                                "Light Fog", "Mist", "Heavy Fog", "Light Haze",

                                                                "Patches of Fog"], "Fog")

ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Scattered Clouds", "Partly Cloudy", 

                                                                 "Mostly Cloudy" ,"Overcast",

                                                                 "Funnel Cloud"], "Cloudy")

ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Light Rain", "Light Drizzle","Rain", "Drizzle", "Light Rain Showers"

                                                                 ,"Drizzle" ,"Rain Showers"], "Rain")

ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Thunderstorms and Rain", "Light Thunderstorms and Rain",

                                                                 "Light Thunderstorm" ,"Heavy Thunderstorms and Rain",

                                                                "Heavy Rain"], "Thunderstorm")

ndls_Temp_org_full["conds"]=ndls_Temp_org_full["conds"].replace(["Thunderstorms with Hail", "Squalls",

                                                                 "Light Hail Showers" ,"Light Freezing Rain",

                                                                "Heavy Thunderstorms with Hail", "Unknown"], "Others")
print("No of unique conditions for GROUPING:",len(ndls_Temp_org_full["conds"].unique()))

plt.figure(figsize=(10,5))

ndls_Temp_org_full["conds"].value_counts().plot(kind="bar")
le=LabelEncoder()

col="conds"

ndls_Temp_org_full[col] = ndls_Temp_org_full.apply(lambda x: le.fit_transform(ndls_Temp_org_full[col].astype(str)), axis=0, result_type='expand')
ndls_Temp_org_full["conds"]
ndls_Temp_org_full["conds"].value_counts()
ndls_Temp_org_full=ndls_Temp_org_full.replace(0, np.nan)
print(ndls_Temp_org_full.isnull().sum())

print("TOTAL NAs:",ndls_Temp_org_full.isnull().sum().sum())
halfrows=0.5*ndls_Temp_org_full.shape[0]

for i in ndls_Temp_org_full.columns:

    totalNA=ndls_Temp_org_full[i].isnull().sum()

    if(totalNA<halfrows):

        if(ndls_Temp_org_full[i].dtypes=="object"):

            tempmode=ndls_Temp_org_full[i].mode()[0]

            print(i," is categorical has",str(totalNA)," NA values replacing is mode: ",tempmode)

            ndls_Temp_org_full[i].fillna(tempmode, inplace=True)

        else:

            tempmedian=ndls_Temp_org_full[i].median()

            print(i," is continuous has",str(totalNA)," NA values replacing is median: ",tempmedian)

            ndls_Temp_org_full[i].fillna(tempmedian, inplace=True)

    else:

        print("Column to drop:", i, "Total NAs", totalNA)

        ndls_Temp_org_full.drop([i], axis=1, inplace=True)
ndls_Temp_org_full
print(ndls_Temp_org_full["wdire"].value_counts())

print("No of unique conditions:",len(ndls_Temp_org_full["wdire"].unique()))
ndls_Temp_org_full["wdire"]=ndls_Temp_org_full["wdire"].replace(["WNW", "WSW", "ESE", "ENE", "NNW", "SSE", "NNE" ,"SSW", "Variable"], 

                                                                ["West", "West", "East", "East", "North", "South", "North", "South", "North"])
ndls_Temp_org_full["wdire"].value_counts()
deg=45

ndls_Temp_org_full["wdire"]=ndls_Temp_org_full["wdire"].replace(["North","NE", "East","SE", "South","SW", "West", "NW"],

                                                                [0, deg, 2*deg, 3*deg, 4*deg, 5*deg, 6*deg, 7*deg])
ndls_Temp_org_full["wdire"].value_counts()
ndls_Temp_org_full
timeseries_fulldata=ndls_Temp_org_full.copy()
timeseries_fulldata["datetimeutc"].dtype
timeseries_fulldata["datetimeutc"]=pd.to_datetime(timeseries_fulldata["datetimeutc"])
timeseries_fulldata.set_index("datetimeutc", inplace=True)
timeseries_fulldata
ndls_daily=timeseries_fulldata.resample("D").mean()
ndls_daily
ndls_daily.isnull().sum() #nulls created due to rollup
ndls_daily.fillna(ndls_daily.mean(), inplace=True)
ndls_daily.isnull().sum()
ndls_daily_temp=pd.DataFrame(list(ndls_daily['tempm']), columns=['temp'])

ndls_daily_temp
plt.figure(figsize=(20,8))

plt.plot(ndls_daily_temp)

plt.grid()

plt.title("Delhi Temp variation (Yearly)") 

plt.show()
scaler=MinMaxScaler(feature_range=(-1,1))

ndls_daily_temp_scaled=scaler.fit_transform(ndls_daily_temp)
print(ndls_daily_temp_scaled)

print(ndls_daily_temp_scaled.shape)
steps=30

X_part=[]

Y_part=[]

for i in range(len(ndls_daily_temp_scaled)-(steps)):

    X_part.append(ndls_daily_temp_scaled[i:i+steps])

    Y_part.append(ndls_daily_temp_scaled[i+steps])

    

X_part=np.array(X_part)

Y_part=np.array(Y_part)



print(X_part.shape)

print(Y_part.shape)
train_X=X_part[:7300,::]

test_X=X_part[7300:,::]

print("train_X Shape:",train_X.shape, ",test_X Shape:", test_X.shape)



train_Y=Y_part[:7300]

test_Y=Y_part[7300:]

print("test_Y Shape:",train_Y.shape, ",test_Y Shape:", test_Y.shape)
model1=models.Sequential()

model1.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu", input_shape=(30,1)))

model1.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu"))

model1.add(layers.MaxPooling1D(pool_size=2))

model1.add(layers.Conv1D(filters=256, kernel_size=2, activation="relu"))

model1.add(layers.Flatten())

model1.add(layers.RepeatVector(30)) #same as input (30,1)

model1.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))

model1.add(layers.Dropout(0.2))

model1.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))

model1.add(layers.Dropout(0.2))

model1.add(layers.Bidirectional(layers.LSTM(units=128, activation="relu")))

model1.add(layers.Dense(100, activation="relu"))

model1.add(layers.Dense(1))
model1.compile(loss="mae", optimizer=optimizers.Adam(lr=0.0001))
model1.summary()
utils.plot_model(model1)
#callbacks

Earlystp=callbacks.EarlyStopping(monitor="loss", mode="min", patience=5, restore_best_weights=True)

Savemod=callbacks.ModelCheckpoint(filepath="model1_ts.h5", monitor="loss", save_best_only=True)
history=model1.fit(train_X, train_Y, epochs=200, verbose=0, callbacks=[Earlystp, Savemod])
hist=history.history

train_loss=hist["loss"]

epoch=range(1,len(train_loss)+1)  #hist is a dict
plt.plot(epoch, train_loss)
model1_pt=models.load_model("model1_ts.h5")
ts_temp=model1_pt.predict(test_X)
ts_temp=scaler.inverse_transform(ts_temp)
ts_temp
test_Y_inv=scaler.inverse_transform(test_Y)
plt.figure(figsize=(20,9))

plt.plot(test_Y_inv , 'blue', linewidth=5)

plt.plot(ts_temp,'r' , linewidth=4)

plt.xlabel("Time", fontsize=20)

plt.ylabel("Temperature (C)", fontsize=20)

plt.legend(('Test','Predicted'))

plt.show()
mse=mean_squared_error(test_Y_inv, ts_temp)

mae=mean_absolute_error(test_Y_inv, ts_temp)

print("Mean Squared Error:", str(mse), "and Mean Absolute Error:", str(mae))
X_part=ndls_daily.drop(["tempm"], axis=1)

Y_part=ndls_daily["tempm"]

X_part=np.array(X_part)

Y_part=np.array(Y_part).reshape(-1,1)
print("X shape:",X_part.shape)

print("Y shape:",Y_part.shape)
scaler2=MinMaxScaler(feature_range=[-1,1])

X_part_scaled=scaler2.fit_transform(X_part)

Y_part_scaled=scaler2.fit_transform(Y_part)
print(X_part)

print("===================Post Scalling====================")

print(X_part_scaled)
print(Y_part)

print("===================Post Scalling====================")

print(Y_part_scaled)
step=30

input=[]

output=[]

for i in range(len(X_part_scaled)-(step)):

    input.append(X_part_scaled[i:i+step])

    output.append(Y_part_scaled[i+step])

 

input=np.array(input)

output=np.array(output)



print(input.shape)

print(output.shape)
trainR_X=input[:7300,::]

testR_X=input[7300:,::]

print("train_X Shape:",trainR_X.shape, ",test_X Shape:", testR_X.shape)



trainR_Y=output[:7300]

testR_Y=output[7300:]

print("test_Y Shape:",trainR_Y.shape, ",test_Y Shape:", testR_Y.shape)
model2=models.Sequential()

model2.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu", input_shape=(30,8)))

model2.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu"))

model2.add(layers.MaxPool1D(pool_size=2))

model2.add(layers.Conv1D(filters=128, kernel_size=2, activation="relu"))

model2.add(layers.Flatten())

model2.add(layers.RepeatVector(30)) #same as input (30,1)=30

model2.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))

model2.add(layers.Dropout(0.2))

model2.add(layers.LSTM(units=100, return_sequences=True, activation="relu"))

model2.add(layers.Dropout(0.2))

model2.add(layers.Bidirectional(layers.LSTM(units=128, activation="relu")))

model2.add(layers.Dense(100, activation="relu"))

model2.add(layers.Dense(1))
utils.plot_model(model2)
model2.compile(optimizer=optimizers.Adam(lr=0.001), loss="mae", metrics=["mse"])
EarlyStp=callbacks.EarlyStopping(monitor="loss", mode="min", patience=5)

Savemod=callbacks.ModelCheckpoint(filepath="model2_R.h5", monitor="loss", save_best_only=True)
history=model2.fit(trainR_X, trainR_Y, epochs=200, verbose=0, callbacks=[Savemod, EarlyStp])
hist=history.history

train_loss=hist["loss"]

epoch=range(1,len(train_loss)+1)  #hist is a dict
plt.plot(epoch, train_loss)
model2=models.load_model("model2_R.h5")
temp_rs=model2.predict(testR_X)
temp_rs=scaler2.inverse_transform(temp_rs)

temp_rs
testR_Y_inv=scaler2.inverse_transform(testR_Y)
plt.figure(figsize=(20,9))

plt.plot(testR_Y_inv , 'blue', linewidth=5)

plt.plot(temp_rs,'r' , linewidth=4)

plt.xlabel("Time", fontsize=20)

plt.ylabel("Temperature (C)", fontsize=20)

plt.legend(('Test','Predicted'))

plt.show()
mse=mean_squared_error(testR_Y_inv, temp_rs)

mae=mean_absolute_error(testR_Y_inv, temp_rs)

print("Mean Squared Error:", str(mse), "and Mean Absolute Error:", str(mae))
final_pred=(temp_rs+ts_temp)/2
plt.figure(figsize=(20,9))

plt.plot(testR_Y_inv , 'red', linewidth=5)

plt.plot(final_pred,'blue' , linewidth=4)

plt.xlabel("Time", fontsize=20)

plt.ylabel("Temperature (C)", fontsize=20)

plt.legend(('Test','Predicted'))

plt.show()
mse=mean_squared_error(testR_Y_inv, final_pred)

mae=mean_absolute_error(testR_Y_inv, final_pred)

print("Mean Squared Error:", str(mse), "and Mean Absolute Error:", str(mae))