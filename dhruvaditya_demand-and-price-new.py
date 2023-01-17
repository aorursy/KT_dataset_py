import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from statsmodels.tsa.seasonal import seasonal_decompose
data=pd.read_csv('../input/nsw-demand/NSW_Demand.csv',index_col='Time', parse_dates=True,dayfirst=True)



demand_decompose = seasonal_decompose(data['Demand (MW)'], model='multiplicative',freq=1)
resid=pd.DataFrame(demand_decompose.resid)
resid=resid.fillna(resid.mean())
data['Demand (MW)']=np.divide(data['Demand (MW)'],list(resid['resid']))

demand_decompose = seasonal_decompose(data['Price (AUD)'], model='additive',freq=1)
resid=pd.DataFrame(demand_decompose.resid)
resid=resid.fillna(resid.mean())
data['Price (AUD)']=data['Price (AUD)']-list(resid['resid'])

# data.head(0)

from matplotlib.pylab import plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse
def mean_square_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


for k,i in enumerate(data['Price (AUD)']):
    if(i<=0):
        data['Price (AUD)'][k]=1                          # Removing the negetive outlyers
    if(i>200):
        data['Price (AUD)'][k]=200
        
for k,i in enumerate(data['Price (AUD)']):
    if(i<0):
        print(i)

data1=np.log(data)        # taking the log of the data 
# data1=data
data1.info()
scaler = MinMaxScaler(feature_range=(-1, 1))     #Scaling the Data
scaler.fit(data1)
all_data=scaler.fit_transform(data1)


# demand_decompose = seasonal_decompose(data['Demand (MW)'], model='multiplicative',freq=50)
# demand_decompose.plot()
# plt.figure()
# # plt.plot(demand_decompose.seasonal[0:100])
# print(len(demand_decompose.resid))
# print(len(data))

train_data=all_data[:int(0.99*(all_data.shape[0]))]
val_data=all_data[int(0.99*(all_data.shape[0])):int(0.998*(all_data.shape[0]))]
test_data=all_data[int(0.998*(all_data.shape[0])):]


t1=data.index[:int(0.99*(all_data.shape[0]))]
t2=data.index[int(0.99*(all_data.shape[0])):int(0.998*(all_data.shape[0]))]
t3=data.index[int(0.998*(all_data.shape[0])):]


source = ColumnDataSource(data)
p = figure(x_axis_type="datetime", plot_width=800, plot_height=800,title='Orignal Data',x_axis_label="Time")
p.line('Time', 'Demand (MW)', source=source,legend_label='Demand in MWH')
p.line('Time', 'Price (AUD)', source=source,color='red',legend_label='Price in AUD$')
# show(p)
print(t3[0])
def preprocess_FNN(data, look_back,return_input_seq=False):
    X_train = []
    y_train = []
    for i in range(data.shape[0]-look_back):
        x1 = data[i:look_back+i,0]
        y1 = data[look_back+i,0]
        x2 = data[i:look_back+i,1]
        y2 = data[look_back+i,1]
        RAND=[]
        X_train.append(RAND)
        for j in range(len(x1)):
            RAND.append([x1[j],x2[j]])        
        y_train.append([y1,y2])

    x1 = data[i+1:look_back+i+1,0]
    x2 = data[i+1:look_back+i+1,1]
    input_seq_for_test=[]
    
    for i in range(len(x1)):
        input_seq_for_test.append([x1[i],x2[i]])
#     input_seq_for_test = [data[i+1:look_back+i+1,0],data[i+1:look_back+i+1,1]] # Last window used for forecating 
    if(return_input_seq):
        return X_train, y_train, input_seq_for_test
    return X_train, y_train
    
look_back=200
# window size
X_train,y_train=preprocess_FNN(train_data,look_back)
X_val,y_val,input_seq_for_test=preprocess_FNN(val_data,look_back,True)

X_train = np.array(X_train)
y_train=np.array(y_train)
X_val = np.array(X_val)
y_val=np.array(y_val)

input_seq_for_test=np.array(input_seq_for_test)
input_seq_for_test=np.reshape(input_seq_for_test,(1,input_seq_for_test.shape[0],input_seq_for_test.shape[1]))
# X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[2],X_train.shape[1]))
print("X_train Shape",X_train.shape)
print("Y_train Shape",y_train.shape)
print("X_val Shape",X_val.shape)
print("Y_val Shape",y_val.shape)
print("Input Sequence for Forecasting Shape ",input_seq_for_test.shape)


from tensorflow import keras
from tensorflow.keras import regularizers
# print(X_train.shape)
# print(y_train.shape)
model = keras.Sequential()
model.add(keras.layers.LSTM(look_back+1, input_shape=(look_back,2),return_sequences=True))
# model.add(keras.layers.LSTM(128,dropout=0.15,return_sequences=True))
# model.add(keras.layers.LSTM(64,return_sequences=True,activation='sigmoid'))
# model.add(keras.layers.LSTM(32,activation='sigmoid'))

model.add(keras.layers.Flatten())


# model.add(keras.layers.Dense(100,activation='tanh'))
# model.add(keras.layers.Dense(50,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
# model.add(keras.layers.Dense(2,activation='tanh',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),bias_regularizer=regularizers.l2(1e-5),
# activity_regularizer=regularizers.l2(1e-5)))
model.add(keras.layers.Dense(2,activation='tanh'))
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train,y_train,epochs=2,shuffle=True,validation_data=(X_val,y_val),batch_size=5000)
# model.fit(X_train,y_train,epochs=2,shuffle=True)

a=model.predict(X_train)
p=figure(x_axis_type="datetime",title="Demand Prediction on Training Data ",x_axis_label="Time",y_axis_label="Demand")
p.line(x=t1[look_back:],y=y_train[:,0],legend_label='Actual')
p.line(x=t1[look_back:],y=a[:,0],color='red',legend_label='Forecasted')
show(p)
p=figure(title="Price Prediction on Training Data ",x_axis_label="Time",y_axis_label="Price")
p.line(x=t1[look_back:],y=y_train[:,1],legend_label='Actual')
p.line(x=t1[look_back:],y=a[:,1],color='red',legend_label='Forecasted')
show(p)

scaled_a=np.exp(scaler.inverse_transform(a))
scaled_y=np.exp(scaler.inverse_transform(y_train))

p=figure(x_axis_type="datetime",title="Demand Prediction on Training Data ",plot_width=800,x_axis_label="Time",y_axis_label="Demand")
p.line(x=t1,y=scaled_y[:,0],legend_label='Actual')
p.line(x=t1,y=scaled_a[:,0],color='red',legend_label='Forecasted')
show(p)
p=figure(x_axis_type="datetime",title="Price Prediction on Training Data ",x_axis_label="Time",y_axis_label="Price")
p.line(x=t1[look_back:],y=scaled_y[:,1],legend_label='Actual')
p.line(x=t1[look_back:],y=scaled_a[:,1],color='red',legend_label='Forecasted')
show(p)
print("Root Mean Squared Error Demand          :",root_mean_squared_error(a[:,0], y_train[:,0]))
print("Root Mean Squared Error Price           :",root_mean_squared_error(a[:,1],y_train[:,1]))
print("Root Mean Squared Error Demand (Scaled) :",root_mean_squared_error(scaled_a[:,0], scaled_y[:,0]))
print("Root Mean Squared Error Price (Scaled)  :",root_mean_squared_error(a[:,1],scaled_y[:,1]))
print() 

p=model.predict(input_seq_for_test)
print(input_seq_for_test.shape)
input_seq_for_test1=np.append(input_seq_for_test[0],p,axis=0)
input_seq_for_test1=np.expand_dims(input_seq_for_test1,axis=0)
print(input_seq_for_test1.shape)
print(model.predict(input_seq_for_test1[:,-look_back:,:]))
print(p.shape)
def forecast_LSTM(model, input_sequence, future_steps,look_back):
    forecasted_values = []
    arr1=[]
    arr2=[]
    
    
    for i in range(future_steps):
        
        forecasted_value = model.predict(input_sequence[:,-look_back:,:])
    
        input_sequence=np.append(input_sequence[0],forecasted_value,axis=0)
        input_sequence=np.expand_dims(input_sequence,axis=0)
        
        arr1.append(forecasted_value[0,0])
        arr2.append(forecasted_value[0,1])
#         print(input_sequence.shape)
        
        
    return arr1,arr2

forecast=100   # Number of Points to Forecast
arr1,arr2=forecast_LSTM(model,input_seq_for_test,forecast,look_back)
aemo_data=pd.read_csv('../input/aemodata1/Forecast 25-11-19 1600.CSV')
aemo_prediction=aemo_data[aemo_data.columns[7]]
aemo_prediction=np.array(list(aemo_prediction[1:390].astype('int')))
aemo_prediction.shape
# print(aemo_prediction)

plt.plot(list(data['Demand (MW)'][t3[0]:][0:50]))
p=figure(x_axis_type="datetime",title="Forecast Demand ",x_axis_label="Time",y_axis_label="Demand")
p.line(x=t3[:forecast],y=test_data[:forecast,0],legend_label='Actual')
p.line(x=t3[:forecast],y=arr1,color='red',legend_label='Forecasted')

show(p)



# plt.title("Price")

p=figure(x_axis_type="datetime",title="Forecast Price ",x_axis_label="Time",y_axis_label="Price")
p.line(x=t3[:forecast],y=test_data[:forecast,1],legend_label='Actual')
p.line(x=t3[:forecast],y=arr2,color='red',legend_label='Forecasted')
show(p)
print("               Calculating the Error on ",forecast,"forecasting steps")
print("Root Mean Squared Error Demand          :",root_mean_squared_error(arr1, test_data[0:forecast,0]))
print("Root Mean Squared Error Price           :",root_mean_squared_error(arr2, test_data[0:forecast,1]))
arr1=np.array(arr1)
arr2=np.array(arr2)

# scaler

arr=np.exp(scaler.inverse_transform(np.array([arr1,arr2]).T))
# print(arr.shape)
arr1=arr[:,0]
arr2=arr[:,1]

scaled_test_data=np.exp(scaler.inverse_transform(np.array(test_data)))
print(scaled_test_data.shape)

print("Root Mean Squared Error Demand          :",root_mean_squared_error(arr1, scaled_test_data[:forecast,0]))
print("Root Mean Squared Error Demand( AEMO)   :",root_mean_squared_error(aemo_prediction[:forecast], scaled_test_data[:forecast,0]))

print("Root Mean Squared Error Price          :",root_mean_squared_error(arr2, scaled_test_data[:forecast,1]))
p=figure(x_axis_type="datetime",title="Forecast Demand on Next  "+str(forecast)+" Time Stamps",x_axis_label="Time",y_axis_label="Demand")
p.line(x=t3[:forecast],y=scaled_test_data[:forecast,0],legend_label='Actual')
# p.line(x=t3[:forecast],y=list(data['Demand (MW)'][t3[0]:][0:forecast]),legend_label='Actual')
p.line(x=t3[:forecast],y=arr1,color='red',legend_label='Forecasted')
# p.line(x=t3[:forecast],y=aemo_prediction[1:len(aemo_data)],color='black',legend_label='AEMO')
show(p)



p=figure(x_axis_type="datetime",title="Forecast Price  on Next  "+str(forecast)+" Time Stamps",x_axis_label="Time",y_axis_label="Price")
p.line(x=t3[:forecast],y=scaled_test_data[:forecast,1],legend_label='Actual')
p.line(x=t3[:forecast],y=arr2,color='red',legend_label='Forecasted')
show(p)
from collections import defaultdict
prediction_value_on_specific_hour_demand = defaultdict(lambda: [])
actual_value_on_specific_hour_demand = defaultdict(lambda: [])
prediction_value_on_specific_hour_price = defaultdict(lambda: [])
actual_value_on_specific_hour_price = defaultdict(lambda: [])

our_prediction_value_on_specific_hour_demand=defaultdict(lambda: [])
aemo_prediction_value_on_specific_hour_demand=defaultdict(lambda: [])
actual_value_on_specific_hour_demand_test=defaultdict(lambda: [])
scaled_a_pd=pd.DataFrame(scaled_a)
scaled_a_pd.index=t1[look_back:]

scaled_y_pd=pd.DataFrame(scaled_y)
scaled_y_pd.index=t1[look_back:]

scaled_test_pd=pd.DataFrame(scaled_test_data[0:len(arr1)])
scaled_test_pd.index=t2[:len(arr1)]
scaled_test_pd=scaled_test_pd.rename(columns={0: "Actual Demand", 1: "Price"})
scaled_test_pd['Our Prediction']=arr1
scaled_test_pd['AEMO Prediction']=aemo_prediction[0:len(arr1)]

# scaled_test_pd


for j,i in enumerate(scaled_test_pd.index):
    
    time=str(i).split(' ')[1]
    our_prediction_value_on_specific_hour_demand[time].append(scaled_test_pd['Our Prediction'][j])
    aemo_prediction_value_on_specific_hour_demand[time].append(scaled_test_pd['AEMO Prediction'][j])
    actual_value_on_specific_hour_demand_test[time].append(scaled_test_pd['Actual Demand'][j])

our_demand_error_over_time=defaultdict(lambda: [])
aemo_demand_error_over_time=defaultdict(lambda: [])
for key in our_prediction_value_on_specific_hour_demand:
    our_demand_error_over_time[key].append(root_mean_squared_error(our_prediction_value_on_specific_hour_demand[key],actual_value_on_specific_hour_demand_test[key]))
    aemo_demand_error_over_time[key].append(root_mean_squared_error(aemo_prediction_value_on_specific_hour_demand[key],actual_value_on_specific_hour_demand_test[key]))
    

error_analyser=pd.DataFrame(index=list(actual_value_on_specific_hour_demand_test.keys()))
x=[]
y=[]
for i in list(our_demand_error_over_time.values()):
    x.append(i)
for i in list(aemo_demand_error_over_time.values()):
    y.append(i)
plt.figure(figsize=(10, 10))
plt.plot(list(actual_value_on_specific_hour_demand_test.keys()),x)
plt.plot(list(actual_value_on_specific_hour_demand_test.keys()),y)
plt.legend(['OUR','AEMO'])
plt.xticks([1,6,12,18,24,30,36,42,47])
plt.title("RMSE Error on Various Times of Day ("+str(forecast)+") Forecasting Stamps")
plt.xlabel('Time Points')
plt.ylabel('Demand')

for j,i in enumerate(scaled_a_pd.index):
    
    time=str(i).split(' ')[1]
    prediction_value_on_specific_hour_demand[time].append(scaled_a_pd[0][j])
    actual_value_on_specific_hour_demand[time].append(scaled_y_pd[0][j])
    
for j,i in enumerate(scaled_a_pd.index):
    time=str(i).split(' ')[1]
    prediction_value_on_specific_hour_price[time].append(scaled_a_pd[1][j])
    actual_value_on_specific_hour_price[time].append(scaled_y_pd[1][j])

price_error_over_time=defaultdict(lambda: [])
demand_error_over_time=defaultdict(lambda: [])
for key in prediction_value_on_specific_hour_demand:
    price_error_over_time[key].append(root_mean_squared_error(prediction_value_on_specific_hour_price[key],actual_value_on_specific_hour_price[key]))
    demand_error_over_time[key].append(root_mean_squared_error(prediction_value_on_specific_hour_demand[key],actual_value_on_specific_hour_demand[key]))

price_arr=[]
demand_arr=[]
keys=[]
for key in price_error_over_time:
    price_arr.append(price_error_over_time[key])
    keys.append(key)
print()
for key in price_error_over_time:
    demand_arr.append(demand_error_over_time[key])
    
dt=pd.DataFrame()
dt['Time']=keys
dt['Price Error']=pd.DataFrame(price_arr)
dt['Demand Error']=pd.DataFrame(demand_arr)
# pd.to_datetime(dt.Time,format='%H:%M:%S').dt.time
dt.index=dt.Time
dt=dt.drop(['Time'],axis=1)
dt.head()

plt.figure(figsize=(10, 10))
plt.plot(dt['Price Error'])
plt.xticks([1,6,12,18,24,30,36,42,47])
plt.legend(['Average RMSE Error'])
plt.xlabel('Time')
plt.ylabel('Average RMSE Error')
plt.title('Average RMSE Error For Price Prediction on Various Time of the Day')
plt.show()



plt.figure(figsize=(10, 10))
plt.plot(dt['Demand Error'])
plt.xticks([1,6,12,18,24,30,36,42,47])
plt.legend(['Average RMSE Error'])
plt.xlabel('Time')
plt.ylabel('Average RMSE Error')
plt.title('Average RMSE Error For Demand Prediction on Various Time of the Day')

plt.show()

