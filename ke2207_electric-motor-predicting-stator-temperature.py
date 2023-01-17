import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import bottleneck as bn # library used for moving average



from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential

from keras.layers import Activation

from keras.optimizers import SGD

from keras.layers import Dense, Dropout,BatchNormalization

from keras.regularizers import l2

from keras.layers import LSTM

from keras.layers import Dropout
# load the dataset into a pandas dataframe

dataset = pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')

dataset.head()
# check if the dataset contains NaN values

dataset.isnull().values.any()
dataset.describe()
# plot the boxplots of all features

plt.tight_layout(pad=0.9)

plt.figure(figsize=(20,15)) 

plt.subplots_adjust(wspace = 0.2)

nbr_columns = 4 

nbr_graphs = len(dataset.columns)

nbr_rows = int(np.ceil(nbr_graphs/nbr_columns)) 

columns = list(dataset.columns.values) 

with sns.axes_style("whitegrid"):

    for i in range(0,len(columns)-1): 

        plt.subplot(nbr_rows,nbr_columns,i+1) 

        ax1=sns.boxplot(x= columns[i], data= dataset, orient="h",color=sns.color_palette("Blues")[3]) 

    plt.show() 
# plotting the Correlation matrix

fig = plt.figure(figsize=(12,9))

sns.heatmap(dataset.corr(),annot=True)

plt.show()
# print the list of all testruns

profile_id_list = dataset.profile_id.unique()

print(profile_id_list)

print("amount of test runs: {0}".format(profile_id_list.size))
# plotting 'stator_yoke','stator_tooth','stator_winding' for a random set of testruns

columns = ['stator_yoke','stator_tooth','stator_winding']

profile_id_list = np.random.choice(profile_id_list, size=8, replace=False)    

nbr_column = 2 

nbr_graph= len(profile_id_list) 

nbr_row = int(np.ceil(nbr_graph/nbr_column)) 

kolomlijst = list(dataset.columns.values) 

plt.figure(figsize=(30,nbr_row*5)) 

    

with sns.axes_style("whitegrid"):    

    for i in range(0,nbr_graph): 

        plt.subplot(nbr_row,nbr_column,i+1) 

        temp = dataset.loc[dataset['profile_id'] == profile_id_list[i]]

        temp = temp.loc[:,columns]

        temp = temp.iloc[::100, :]

        ax1=sns.lineplot(data=temp.loc[:,columns], 

                        dashes = False,

                        palette=sns.color_palette('Dark2',n_colors=len(columns)))

        ax1.set_title("profile id: {0}".format(profile_id_list[i]))

    plt.show    
# plotting the stator winding temperature in comparison to torque and motorspeed

profile_id = 6

feat_plot_1 = ['stator_winding']

feat_plot_2 = ['torque','motor_speed']

temp = dataset.loc[dataset['profile_id'] == profile_id]

temp = temp.iloc[::10, :]



with sns.axes_style("whitegrid"):

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(211)

    ax1 = sns.lineplot(data=temp.loc[:,feat_plot_1], dashes = False,

                       palette=sns.color_palette('Dark2',n_colors=len(feat_plot_1)),linewidth=0.8)

    ax2 = fig.add_subplot(212)

    ax2 = sns.lineplot(data=temp.loc[:,feat_plot_2], dashes = False,

                       palette=sns.color_palette('Dark2',n_colors=len(feat_plot_2)),linewidth=0.8)

    plt.show()
# Seperating input and output variables

X = dataset.drop('torque', axis=1).loc[:,'ambient':'i_q'].values 

y = dataset.loc[:,'stator_winding'].values 



# split up in training and test data

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)
X.shape
# training the Random Forest Regressor on the dataset

from sklearn.ensemble import RandomForestRegressor

RFR_model = RandomForestRegressor(n_estimators = 10, random_state = 0)

RFR_model.fit(X_train, y_train)
# Calculate MSE and MAE for the entire testset

y_pred = RFR_model.predict(X_test)

RFR_MSE = mean_squared_error(y_test, y_pred)

RFR_MAE = mean_absolute_error(y_test, y_pred)

print("MSE: {0}".format(RFR_MSE))

print("MAE: {0}".format(RFR_MAE))
# plot the true vs predicted values for multiple testruns

test_run_list = np.array([27,45,60,74])

#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]

output_value = 'stator_winding'

model = RFR_model



with sns.axes_style("whitegrid"):    

    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)

    

    for i in range(0,len(test_run_list)):

        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 

        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 

        y_pred_plot = model.predict(X_plot)



        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)

        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)

        axs[i,0].legend(loc='best')

        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))

    plt.show()   
# This is the testrun we will use as an example for the evaluation of the models

# change this value to use a different testrun

choosen_example_testrun = 76
# plot the true vs predicted values for a choosen testrun without and with moving average smoothing:

profile_id = choosen_example_testrun

output_value = 'stator_winding'

model = RFR_model

moving_average_window = 100



X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == profile_id,'ambient':'i_q'].values 

y_plot = dataset.loc[dataset['profile_id'] == profile_id,output_value].values 

y_pred_plot = model.predict(X_plot)

y_pred_plot_smooth = bn.move_mean(y_pred_plot,moving_average_window,1)

time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])



with sns.axes_style("whitegrid"):

    fig = plt.figure(figsize=(20, 10))

    

    ax1 = fig.add_subplot(211)

    ax1.plot(time,y_pred_plot,label='predict without smoothing',color='red',alpha=0.4,linewidth=0.8)

    ax1.plot(time,y_plot,label='True',color='black',linewidth=1)

    ax1.legend(loc='best')

    ax1.set_title("profile id: {0} without smoothing".format(profile_id))

    

    ax2 = fig.add_subplot(212)

    ax2.plot(time,y_pred_plot_smooth,label='predict with smoothing',color='red',alpha=0.8,linewidth=0.8)

    ax2.plot(time,y_plot,label='True',color='black',linewidth=1)

    ax2.legend(loc='best')

    ax2.set_title("profile id: {0} with smoothing".format(profile_id))

    

    plt.show()



# Calculate MSE and MAE for the choosen testrun without and with moving average smoothing:

MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot)

MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot)

print("metrics without moving average smoothing:")

print("MSE: {0}".format(MSE_RFR_model))

print("MAE: {0}".format(MAE_RFR_model))

MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot_smooth)

MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot_smooth)

print("metrics with moving average smoothing:")

print("MSE: {0}".format(MSE_RFR_model))

print("MAE: {0}".format(MAE_RFR_model))
# constructing and training the neural network

nr_epochs=200

b_size=1000



NN_reg_model = Sequential()

NN_reg_model.add(Dense(11, input_dim=X_train.shape[1], activation='relu'))

NN_reg_model.add(Dense(9, activation='relu'))

NN_reg_model.add(Dense(7, activation='relu'))

NN_reg_model.add(Dense(5, activation='relu'))

NN_reg_model.add(Dense(1))

NN_reg_model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["mean_squared_error"])

history = NN_reg_model.fit(X_train, y_train, validation_split=0.33,epochs=nr_epochs, batch_size=b_size, verbose=0)
#plot the history of the model accuracy during training

plt.figure(figsize=(18,6))

ax1=plt.subplot(1, 2, 1)

ax1=plt.plot(history.history['mean_squared_error'],color='blue')

ax1=plt.plot(history.history['val_mean_squared_error'],color='red',alpha=0.5)

ax1=plt.title('model accuracy')

ax1=plt.ylabel('accuracy')

ax1=plt.xlabel('epoch')

ax1=plt.legend(['train', 'test'], loc='upper left')



# plot the history of the model loss during training

ax2=plt.subplot(1, 2, 2)

ax2=plt.plot(history.history['loss'],color='blue')

ax2=plt.plot(history.history['val_loss'],color='red',alpha=0.5)

ax2=plt.title('model loss')

ax2=plt.ylabel('loss')

ax2=plt.xlabel('epoch')

ax2=plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Calculate MSE and MAE of the entire testset

y_pred = NN_reg_model.predict(X_test)

NN_MSE = mean_squared_error(y_test, y_pred)

NN_MAE = mean_absolute_error(y_test, y_pred)

print("MSE: {0}".format(NN_MSE))

print("MAE: {0}".format(NN_MAE))
# plot the true vs predicted values for multiple testruns

test_run_list = np.array([27,45,60,74])

#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]

output_value = 'stator_winding'

model = NN_reg_model



with sns.axes_style("whitegrid"):    

    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)

    

    for i in range(0,len(test_run_list)):

        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 

        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 

        y_pred_plot = model.predict(X_plot)



        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)

        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)

        axs[i,0].legend(loc='best')

        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))

    plt.show()   
# plot the true vs predicted values for a choosen testrun without and with moving average smoothing:

profile_id = choosen_example_testrun

output_value = 'stator_winding'

model = NN_reg_model

moving_average_window = 100



X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == profile_id,'ambient':'i_q'].values 

y_plot = dataset.loc[dataset['profile_id'] == profile_id,output_value].values 

y_pred_plot = model.predict(X_plot).flatten()

y_pred_plot_smooth = bn.move_mean(y_pred_plot,moving_average_window,1)

time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])



with sns.axes_style("whitegrid"):

    fig = plt.figure(figsize=(20, 10))

    

    ax1 = fig.add_subplot(211)

    ax1.plot(time,y_pred_plot,label='predict without smoothing',color='red',alpha=0.4,linewidth=0.8)

    ax1.plot(time,y_plot,label='True',color='black',linewidth=1)

    ax1.legend(loc='best')

    ax1.set_title("profile id: {0} without smoothing".format(profile_id))

    

    ax2 = fig.add_subplot(212)

    ax2.plot(time,y_pred_plot_smooth,label='predict with smoothing',color='red',alpha=0.8,linewidth=0.8)

    ax2.plot(time,y_plot,label='True',color='black',linewidth=1)

    ax2.legend(loc='best')

    ax2.set_title("profile id: {0} with smoothing".format(profile_id))

    

    plt.show()



# Calculate MSE and MAE for the choosen testrun without and with moving average smoothing:

MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot)

MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot)

print("metrics without moving average smoothing:")

print("MSE: {0}".format(MSE_RFR_model))

print("MAE: {0}".format(MAE_RFR_model))

MSE_RFR_model = mean_squared_error(y_plot, y_pred_plot_smooth)

MAE_RFR_model = mean_absolute_error(y_plot, y_pred_plot_smooth)

print("metrics with moving average smoothing:")

print("MSE: {0}".format(MSE_RFR_model))

print("MAE: {0}".format(MAE_RFR_model))
# plot the amount of samples per testrun

plt.figure(figsize=(18,5))

sns.countplot(x="profile_id", data=dataset,color=sns.color_palette("Blues")[2])

plt.show()
# find the testrun with the least amount of samples

max_batch_size = dataset.profile_id.value_counts().min()

profile_id_list = dataset.profile_id.unique()

print("list of testruns:")

print(profile_id_list)

print("smallest amount of samples in one testrun: {0}".format(max_batch_size))

print("testrun with smallest amount of samples: {0}".format(dataset.profile_id.value_counts().idxmin()))
# function to create time-step windows for LSTM



def sliding_window(profile_id_list,max_sample_count,sample_rate=1,window_size=100):

    # profile_id_list: list of testruns we want to use to extract our samples

    # max_sample_count: the total amount of samples we want in our trainingset

    # sample rate: amount of samples to skip between the previous and next sample

    # window_size: amount of time steps (samples in the past) the window contains

    

    nr_of_features = 7 #number of columns minus 'stator_winding','profile_id'

    sample_count = 0



    i = 0

    X = np.zeros((max_sample_count,window_size,nr_of_features))

    y = np.zeros((max_sample_count))



    for profile_id in profile_id_list:

        temp=(dataset[dataset['profile_id']==profile_id]).iloc[lambda x: x.index % sample_rate==0]     

        temp_y = temp['stator_winding']

        temp_X = temp.drop('torque', axis=1).loc[:,'ambient':'i_q']

    

        i=0

        while i < len(temp_X)-window_size and sample_count < max_sample_count:

            X[sample_count] = temp_X.iloc[i:i+window_size]

            y[sample_count] = temp_y.iloc[i+window_size]

            sample_count +=1

            i +=1

    return (X,y) 

        
# split the testruns in a training and testset

profile_id_list_train ,profile_id_list_test = train_test_split(profile_id_list,test_size=0.3)

print("the list of testruns used for extracting the training sample windows:")

print(profile_id_list_train)

print("the list of testruns used for extracting the testing sample windows:")

print(profile_id_list_test)
# constructing and training the LSTM

window_Size = 100

sample_amount = 5000

sample_rate = 10

epoch= 200

b_size = 500



X_train, y_train = sliding_window(profile_id_list_train,sample_amount,sample_rate,window_Size)

X_test, y_test = sliding_window(profile_id_list_test,sample_amount,sample_rate,window_Size)



LSTM_model = Sequential()

LSTM_model.add(LSTM(128, input_shape = (window_Size,7),return_sequences=True))

LSTM_model.add(LSTM(64, return_sequences=False))

LSTM_model.add(Dense(32, activation='relu'))

LSTM_model.add(Dropout(0.2))

LSTM_model.add(Dense(16, activation='relu'))

LSTM_model.add(Dense(8, activation='relu'))

LSTM_model.add(Dense(1))

LSTM_model.compile(loss='mean_squared_error', optimizer='adam',metrics = ["mean_squared_error"])

history = LSTM_model.fit(X_train,y_train,validation_split=0.33, epochs = epoch, batch_size = b_size, verbose = 0)
#plot the history of the model accuracy during training

plt.figure(figsize=(18,6))

ax1=plt.subplot(1, 2, 1)

ax1=plt.plot(history.history['mean_squared_error'],color='blue')

ax1=plt.plot(history.history['val_mean_squared_error'],color='red',alpha=0.5)

ax1=plt.title('model accuracy')

ax1=plt.ylabel('accuracy')

ax1=plt.xlabel('epoch')

ax1=plt.legend(['train', 'test'], loc='upper left')



# plot the history of the model loss during training

ax2=plt.subplot(1, 2, 2)

ax2=plt.plot(history.history['loss'],color='blue')

ax2=plt.plot(history.history['val_loss'],color='red',alpha=0.5)

ax2=plt.title('model loss')

ax2=plt.ylabel('loss')

ax2=plt.xlabel('epoch')

ax2=plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Calculate MSE and MAE of the entire testset

y_pred_LSTM = LSTM_model.predict(X_test)

LSTM_MSE = mean_squared_error(y_test, y_pred_LSTM)

LSTM_MAE = mean_absolute_error(y_test, y_pred_LSTM)

print("MSE: {0}".format(LSTM_MSE))

print("MAE: {0}".format(LSTM_MAE))
# plot the true vs predicted values for multiple testruns

test_run_list = np.array([27,45,60,74])

#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]

output_value = 'stator_winding'

model = LSTM_model



window_Size = 100

sample_rate = 10



with sns.axes_style("whitegrid"):    

    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)

    

    for i in range(0,len(test_run_list)):

        sample_amount = (len(dataset[dataset['profile_id']==test_run_list[i]])-window_Size)//sample_rate

        X_plot, y_plot = sliding_window([test_run_list[i]],sample_amount,sample_rate,window_Size)

        y_pred_plot = model.predict(X_plot)



        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)

        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)

        axs[i,0].legend(loc='best')

        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))

    plt.show()   
# plot the true vs predicted values for one run

test_run_list = np.array([choosen_example_testrun])

#test_run_list = np.random.choice(profile_id_list, size=4, replace=False)]

output_value = 'stator_winding'

model = LSTM_model



window_Size = 100

#sample_amount = 10000

sample_rate = 10



with sns.axes_style("whitegrid"):    

    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)

    

    for i in range(0,len(test_run_list)):

        sample_amount = (len(dataset[dataset['profile_id']==test_run_list[i]])-window_Size)//sample_rate

        X_plot, y_plot = sliding_window([test_run_list[i]],sample_amount,sample_rate,window_Size)

        y_pred_plot = model.predict(X_plot)



        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)

        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)

        axs[i,0].legend(loc='best')

        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))

    plt.show()

    

print("MSE: {0}".format(mean_squared_error(y_plot, y_pred_plot)))

print("MAE: {0}".format(mean_absolute_error(y_plot, y_pred_plot)))
# plot the true vs predicted values for multiple testruns for the Random Forest Regressor

#test_run_list = np.array([27,45,60,74])

test_run_list = np.random.choice(profile_id_list, size=4, replace=False)

output_value = 'stator_winding'

model = RFR_model

moving_average = 100



with sns.axes_style("whitegrid"):    

    fig, axs = plt.subplots(len(test_run_list),1,figsize=(20,len(test_run_list)*5),squeeze=False)

    

    for i in range(0,len(test_run_list)):

        X_plot = dataset.drop('torque', axis=1).loc[dataset['profile_id'] == test_run_list[i],'ambient':'i_q'].values 

        y_plot = dataset.loc[dataset['profile_id'] == test_run_list[i],output_value].values 

        y_pred_plot = bn.move_mean(model.predict(X_plot),moving_average_window,1)



        time = np.linspace(0, y_plot.shape[0],num=y_plot.shape[0])

        axs[i,0].plot(time,y_pred_plot,label='predict',color='red',alpha=0.4,linewidth=0.8)

        axs[i,0].plot(time,y_plot,label='True',color='black',linewidth=1)

        axs[i,0].legend(loc='best')

        axs[i,0].set_title("profile id: {0}".format(test_run_list[i]))

    plt.show()   