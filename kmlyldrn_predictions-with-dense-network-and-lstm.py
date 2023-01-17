# Necessary Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import cv2

import sklearn

import keras



from sklearn.preprocessing import MinMaxScaler

from datetime import datetime, date

from keras import Sequential

from keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/earthquakes/Earthquakes.csv', delimiter=',')
data.head()
data.describe()
# Function to calculate the days passed

def calculate_days(date_string_1, date_string_2):

  date_num_1=datetime.strptime(date_string_1, '%Y-%m-%d %H:%M:%S')

  date_num_2=datetime.strptime(date_string_2, '%Y-%m-%d %H:%M:%S')



  # Sorting the dates descending

  if date_num_2>date_num_1:

    a=date_num_1

    date_num_1=date_num_2

    date_num_2=a



  # Concatenating dates

  d1=date(year=date_num_1.year, month=date_num_1.month, day=date_num_1.day)

  d2=date(year=date_num_2.year, month=date_num_2.month, day=date_num_2.day)



  return (d1-d2).days



#-------------------------------------------------------------------------



# Function to scale the time between [0,1)

def hour_rate(date_string):

  date_num=datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')



  # Calculating the rate

  hr=((date_num.hour*60) + date_num.minute) / (24*60)



  return hr
data["Time Gap"]=[calculate_days(data["Date(UTC)"][i], data["Date(UTC)"][i-1]) if i>0 else 0 for i in range(0, len(data["Date(UTC)"]))]



data["Hour-Day Ratio"]=[hour_rate(data["Date(UTC)"][i]) for i in range(0, len(data["Date(UTC)"]))]
begin_date="1900-01-01 00:00:00"



data["Days"]=[calculate_days(data["Date(UTC)"][i], begin_date) for i in range(0, len(data["Date(UTC)"]))]



data["Month"]=[datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month for i in data["Date(UTC)"]]
data["Constant Deg."]=data["Constant Deg."].replace({"No":0, "Yes":1})



data=data.sort_values(by="Days", ascending=True).reset_index(drop=True)



data=data.drop(["No", "Ref1", "Source Description 1", "Source No 2", 

                "Source Description 2", "Source No 3", "Source Description 3", 

                "Type", "Date(UTC)"], axis=1)
# Loading risk map

raw_img=cv2.imread("/kaggle/input/earthquakes/risk_map_clean.jpg", cv2.IMREAD_GRAYSCALE)
# Recovering process



# Initial values

size=6

increment=2

epoch=4

recovered_img=raw_img.copy()



# Filtering

for i in range(0,epoch):

  

  width_step=np.shape(recovered_img)[1]/size

  height_step=np.shape(recovered_img)[0]/size



  # Filter Striding

  for h in range(0, int(height_step)):

    for w in range(0, int(width_step)):



      window=recovered_img[h*size:(h+1)*size, w*size:(w+1)*size]



      # At first epoch, values are maximized; then minimized. 

      if i==0:

        window=window.max()

      else:

        window=window.min()



      recovered_img[h*size:(h+1)*size, w*size:(w+1)*size]=window



  size+=increment
# Value Replacement

risk_map=recovered_img.copy()



# Threshold values

high=90

medium=175

low=235

no_data=250

default=5



# Replacement

risk_map=np.where(risk_map<=high, 4, risk_map)

risk_map=np.where(((risk_map>high) & (risk_map<=medium)), 3, risk_map)

risk_map=np.where(((risk_map>medium) & (risk_map<=low)), 2, risk_map)

risk_map=np.where(((risk_map>low) & (risk_map<=no_data)), 1, risk_map)

risk_map=np.where(risk_map>no_data, default, risk_map)
# Visualization of all 3 maps

map_names={"Raw Risk Map":raw_img, "Recovered Risk Map":recovered_img, "Ready-to-Use Risk Map":risk_map}



fig=plt.figure(figsize=(16, 9))



for i, val in enumerate(map_names):

    fig.add_subplot(2, 2, i+1)

    plt.imshow(map_names[val], cmap="gray")

    plt.title(val)

    plt.tight_layout()



plt.show()
#Function to find risk class

def risk_grader(latitude, longitude):

  # Begin and end coordinates of the map used.

  west=25.67

  east=44.81

  south=35.81

  north=42.10



  # Checking coordinates whether involved by map

  if (longitude<west) or (longitude>east) or (latitude<south) or (latitude>north):

    return default



  # Calculating ratio between real land piece and map image pixels

  real_width=east-west

  real_height=north-south



  map_width=np.shape(risk_map)[1]

  map_height=np.shape(risk_map)[0]



  width_ratio=map_width/(real_width*100)

  height_ratio=map_height/(real_height*100)



  # Calculating pixels to look up for the grade

  easting=longitude-west

  northing=latitude-south



  pixel_to_right=int(round(easting*100*width_ratio))

  pixel_to_up=map_height-int(round(northing*100*height_ratio))



  # Correction of the error caused by floating points

  if pixel_to_right>=map_width:

    pixel_to_right=map_width-1



  if pixel_to_up>=map_height:

    pixel_to_up=map_height-1



  # reading risk grade from the map array

  grade=risk_map[pixel_to_up, pixel_to_right]



  return grade
# Finding risk grade for every earthquake

data["Risk Grade"]=[risk_grader(data["Latitude"][i], data["Longitude"][i]) for i in range(len(data["Latitude"]))]
# A glance on dataset

data.head()
# Calculating and concatenating

extended_data=pd.DataFrame()



extended_data=data.copy()



for i in extended_data.columns:

  extended_data["Log."+i]=np.log(extended_data[i]+0.01)
# Scatter Plots

lines=(((len(extended_data.columns)-1)*(len(extended_data.columns)))/8)+1

k=1



subplt=plt.figure(figsize=(16, 120))



for i in range(0, len(extended_data.columns)-1):

  for j in range(i+1, len(extended_data.columns)-1):

    subplt.add_subplot(lines, 4, k)

    

    plt.scatter(extended_data[extended_data.columns[i]], extended_data[extended_data.columns[j]])

    plt.title("{} X {}".format(extended_data.columns[i], extended_data.columns[j]))

    plt.xlabel(extended_data.columns[i])

    plt.ylabel(extended_data.columns[j])



    k+=1



plt.tight_layout()

plt.show()
extended_corr=extended_data.corr()



plt.figure(figsize=(20,9))

sns.heatmap(extended_corr, vmin=-1, vmax=1, cmap="bwr", annot=True, linewidth=0.1)

plt.title("Parametre Correlation Matrix")

plt.show()
new_labels=["Latitude", "Longitude", "Days", "Magnitude", 

            "Depth", "Constant Deg.", "Risk Grade", "Time Gap", 

            "Log.Days", "Log.Hour-Day Ratio"]



new_data=extended_data[new_labels]
percentage=0.25



duration=20



test_size=int(len(new_data)*percentage)
# Dense Network Dataset partition

per=0.25



test_size=int(len(new_data)*per)



new_train=new_data[0:len(new_data)-test_size]

new_test=new_data[len(new_data)-test_size:len(new_data)]



X_train_Dense=new_train.iloc[:,4:]

X_test_Dense=new_test.iloc[:,4:]



y_train_Dense=new_train.iloc[:,0:4]

y_test_Dense=new_test.iloc[:,0:4]



# Data set scaling

scaler_Dense_X=MinMaxScaler(feature_range=(0,1))

scaler_Dense_y=MinMaxScaler(feature_range=(0,1))



X_train_Dense=scaler_Dense_X.fit_transform(X_train_Dense)

X_test_Dense=scaler_Dense_X.fit_transform(X_test_Dense)



y_train_Dense=scaler_Dense_y.fit_transform(y_train_Dense)

y_test_Dense=scaler_Dense_y.fit_transform(y_test_Dense)
# Dense Network Dataset Scaling and Partition



# Definings

scaler_LSTM_X=MinMaxScaler(feature_range=(0,1))

scaler_LSTM_y=MinMaxScaler(feature_range=(0,1))



new_data_array=np.array(new_data)



X_LSTM, y_LSTM=[], []



# y Subset Seperation and Scaling

y_LSTM=new_data_array[duration:,0:4]

y_LSTM=scaler_LSTM_y.fit_transform(y_LSTM)



# X Subset Seperation an Scaling

new_data_array=scaler_LSTM_X.fit_transform(new_data_array)



for i in range(0,len(new_data_array)-duration):

  partial=new_data_array[i:i+duration]

  X_LSTM=np.append(X_LSTM, partial)



X_LSTM=np.reshape(X_LSTM, (np.shape(new_data_array)[0]-duration, duration, np.shape(new_data_array)[1]))



# Partition to Train and Test Subsets

X_train_LSTM=X_LSTM[0:len(X_LSTM)-test_size]

X_test_LSTM=X_LSTM[len(X_LSTM)-test_size:]



y_train_LSTM=y_LSTM[0:len(y_LSTM)-test_size]

y_test_LSTM=y_LSTM[len(y_LSTM)-test_size:]
def loss_reduce(loss):

  new_loss=[]

  fold=int(len(loss)/np.min(epoch))



  for i in range(0,np.min(epoch)):

    local_mean=np.mean(loss[fold*i:fold*(i+1)])

    new_loss.append(local_mean)



  return new_loss
# Building preset models

def model_builder(builder="Dense", hidden=32, optimizer="rmsprop"):

  if builder=="Dense":

    model=Sequential()



    model.add(Dense(hidden, activation="relu", input_shape=(6,)))

    model.add(BatchNormalization())

    model.add(Dense(4))



    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])



  if builder=="LSTM":

    model=Sequential()



    model.add(GRU(hidden, return_sequences=True))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(LSTM(units=hidden))

    model.add(Dense(units=4))



    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])



  return model
# Searching of hyperparameters

def GraphSearch(model_type):



  # Getting beginning values

  files=False

  hid, opt, bat, epo, row=0, 0, 0, 0, 0

  min_score=1000000

  best_model=[]

    

  loss=pd.DataFrame()

  values=pd.DataFrame()

    

  # Loading relevant dataset

  if model_type=="LSTM":

    X_train=X_train_LSTM

    X_test=X_test_LSTM

    y_train=y_train_LSTM

    y_test=y_test_LSTM



  else:

    X_train=X_train_Dense

    X_test=X_test_Dense

    y_train=y_train_Dense

    y_test=y_test_Dense



  # Graph searching

  for hid in hidden:

    for opt in optimizers:

      for bat in batch:

        for epo in epoch:



          print("{} / {} ".format(row+1, len(hidden)*len(optimizers)*len(batch)*len(epoch)))



          # Training

          model=model_builder(builder=model_type, hidden=hid, optimizer=opt)



          record=model.fit(X_train, y_train, epochs=epo, batch_size=bat, validation_split=0.25, verbose=0)



          # Evaluating and collecting records

          evaluation=model.evaluate(X_test, y_test)



######     loss[row]=loss_reduce(record.history["loss"])

          loss=loss.append(loss_reduce(record.history["loss"]))



          values=values.append({"hidden":hid,

                                "optimizer":opt,

                                "batch":bat,

                                "epochs":epo,

                                "evaluation_0":evaluation[0],

                                "evaluation_1":evaluation[1]},

                                ignore_index=True)



            # Fixing the best model

          if evaluation[0]<min_score:

            min_score=evaluation[0]

            best_model=model



          row+=1



          print("Done!..\n")

            

  return loss, values, best_model
# Parameters and running of dense model

hidden=[32, 64]

optimizers=["rmsprop"]

batch=[32, 64]

epoch=[300, 700]



loss_Dense, values_Dense, model_Dense=GraphSearch("Dense")
# Parameters and running of LSTM model

hidden=[30]

optimizers=["rmsprop"]

batch=[32, 64, 128]

epoch=[750]



loss_LSTM, values_LSTM, model_LSTM=GraphSearch("LSTM")
# Definings for easy use

variables=["epochs", "batch", "optimizer", "hidden"]



types=["Dense", "LSTM"]



dict={"values_Dense":values_Dense,

      "values_LSTM":values_LSTM,

      "loss_Dense":loss_Dense,

      "loss_LSTM":loss_LSTM}
# Models 

sns.set(style="darkgrid")

fig=plt.figure(figsize=(16, 7))

i=1



plt.suptitle("Min - Mean - Max Values on Each Hyperparameter", y=1.03)

for val1 in types:

  for val2 in variables:

    fig.add_subplot(1, 8, i)



    a=dict["values_"+val1].groupby(val2)["evaluation_0"].max()

    sns.barplot(x=a.index, y=a.values)

    b=dict["values_"+val1].groupby(val2)["evaluation_0"].mean()

    sns.barplot(x=b.index, y=b.values)

    c=dict["values_"+val1].groupby(val2)["evaluation_0"].min()

    sns.barplot(x=c.index, y=c.values)



    plt.title(val1+" Network")

    plt.xticks(rotation=45)

    plt.tight_layout()

    i+=1

plt.show()
values_Dense[values_Dense["evaluation_0"]==values_Dense["evaluation_0"].min()]
values_LSTM[values_LSTM["evaluation_0"]==values_LSTM["evaluation_0"].min()]
variables=["epochs", "batch", "optimizer", "hidden"]



col="batch" # Change this and run again!



count=0

fig=plt.figure(figsize=(16, 8))



for i in types:

  mod_type=i

  s1="loss_"+mod_type

  s2="values_"+mod_type



  fig.add_subplot(2, 2, count*2+1)

  sns.lineplot(x=dict[s1].index, y=dict[s1].T.min())

  plt.title("Minimum Values ( "+i+" )")

  plt.legend(dict[s2][col].unique())



  fig.add_subplot(2, 2, count*2+2)

  sns.lineplot(x=dict[s1].index, y=dict[s1].T.max())

  plt.title("Maximum Values ( "+i+" )")

  plt.legend(dict[s2][col].unique())



  count+=1



plt.tight_layout()

plt.show()
pred_Dense=model_Dense.predict(X_test_Dense)

pred_LSTM=model_LSTM.predict(X_test_LSTM)



pred_Dense=scaler_Dense_y.inverse_transform(pred_Dense)

pred_LSTM=scaler_LSTM_y.inverse_transform(pred_LSTM)

real_values=scaler_Dense_y.inverse_transform(y_test_Dense)
plt.figure(figsize=(20,6))

plt.scatter(real_values.T[1], real_values.T[0], label="Real Values", alpha=0.6)

plt.scatter(pred_Dense.T[1], pred_Dense.T[0], label="Dense Prediction", alpha=0.6)

plt.scatter(pred_LSTM.T[1], pred_LSTM.T[0], label="LSTM Prediction", alpha=0.4)

plt.title("Coordinates Prediction")

plt.legend()

plt.show()
ind=[i for i in range(0, len(real_values))]



plt.figure(figsize=(20,6))

plt.scatter(ind, real_values.T[2], label="Real Values")

plt.scatter(ind, pred_Dense.T[2], label="Dense Prediction")

plt.scatter(ind, pred_LSTM.T[2], label="LSTM Prediction")

plt.title("Days Prediction")

plt.legend()

plt.show()
plt.figure(figsize=(20,6))

plt.scatter(ind, real_values.T[3], label="Real Values")

plt.scatter(ind, pred_Dense.T[3], label="Dense Prediction")

plt.scatter(ind, pred_LSTM.T[3], label="LSTM Prediction")

plt.title("Magnitude Prediction")

plt.legend()

plt.show()