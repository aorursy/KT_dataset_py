!pip install tslearn
"""
Predict next 5 min
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv
from tslearn.clustering import KShape
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
import gc

"""Convert an array of values into a dataset matrix"""

def create_dataset(dataset, look_back):
    dataX = []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back) + 1, 0]
        dataX.append(a)
    return np.array(dataX) # [x1,x2,x3,y]

def create_dataset_fifteen(dataset, look_back):
    dataX = []
    for i in range(len(dataset) - look_back - 2):
        a = dataset[i:(i + look_back), 0] # past three timestamp
        b = dataset[i + look_back + 2, 0] # next 15 min
        c = np.concatenate((a, b), axis=None)
        dataX.append(c)
    return np.array(dataX) # [x1,x2,x3,y]

def create_dataset_thirty(dataset, look_back):
    dataX = []
    for i in range(len(dataset) - look_back - 5):
        a = dataset[i:(i + look_back), 0] # past three timestamp
        b = dataset[i + look_back + 5, 0] # next 15 min
        c = np.concatenate((a, b), axis=None)
        dataX.append(c)
    return np.array(dataX) # [x1,x2,x3,y]

"""
Compute average value of a list 
  @ Input: list

  @ Output: average value of the list
"""


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


"""K-shape clustering"""
vector_df = pd.read_csv('../input/vector_Nov.csv', dtype=np.float16)
X = vector_df.drop(columns=['Unnamed: 0']).values  # The occupied probability in every 5 minutes

del vector_df
gc.collect()

parking_df = pd.read_csv('../input/Parking_Nov.csv',usecols=['DeviceId', 'ArrivalTime', 'DepartureTime'])
device = parking_df['DeviceId'].unique()

final_results = []
with open("KShape_LSTM_Nov_15min.csv", "a") as csvfile:
    csvfile.write('n_clusters,TrainRMSE,TestRMSE' + '\n')
    for nums in range(9,11):
        start_time = time.time()
        ks = KShape(n_clusters=nums, n_init=10, verbose=False, random_state=3)
        labels = ks.fit_predict(X)
        kshape_label = np.vstack((device, labels))
        kshape_label_df = pd.DataFrame(kshape_label, dtype=np.int16)
        # print(kshape_label_df.head())

        """Combine label and deviceID"""
        dataset2 = kshape_label_df.T

        a = []
        for j in parking_df['DeviceId'][:]:
            b = dataset2[dataset2[0] == j]
            c = np.array(b)
            a.append(c[0][1])

        d = {'Group': a}
        label_df = pd.DataFrame(data=d)
        frames = [parking_df, label_df]
        kshape_group_df = pd.concat(frames, axis=1)
        
#         del parking_df
#         gc.collect()
        # kshape_group_df.to_csv('one_month_group_k7.csv',index=False)


        """Compute occupancy rate"""
        start_day=1
        end_day=30
        start_hour=0
        end_hour=24
        sample_min=5
        
        # group = np.array(kshape_group_df.Group.unique())
        group = kshape_group_df.Group.unique().tolist()
        # total = []
        total_occ = np.empty([nums,(end_day-start_day+1)*(end_hour-start_hour)*12],dtype=np.float16)
        kshape_group_df['DepartureTime'] = pd.to_datetime(kshape_group_df['DepartureTime'])
        kshape_group_df['ArrivalTime'] = pd.to_datetime(kshape_group_df['ArrivalTime'])

        for s in group:
            print(group.index(s))
            cluster_df = kshape_group_df.loc[kshape_group_df['Group'] == s]
            df2 = cluster_df.DeviceId.unique()  # DeviceId in one cluster
            number = len(df2)  # Number of devices in one cluster
            departure_df = cluster_df['DepartureTime']

            for d in range(start_day, end_day+1):
                for h in range(start_hour, end_hour):
                    for min in range(0, 60, 5):
                        parked = 0
                        sample_time = datetime(2011, 11, d, h, min, 0)
                        late_df = cluster_df[departure_df >= sample_time]
                        early_df = late_df[late_df['ArrivalTime'] < sample_time]
                        parked = early_df.shape[0]
                        cols = (d-start_day)*(end_hour-start_hour)*12 + (h-start_hour)*12 + min/sample_min
                        cols=int(cols)
                        total_occ[group.index(s),cols] = (float(parked) / number)  # occ rate in every 5 minutes
        occupancy_kmeans_df = pd.DataFrame(total_occ)
        del kshape_group_df
        gc.collect()

        """LSTM on every cluster"""

        # fix random seed for reproducibility
        np.random.seed(7)

        train_score_list = []
        test_score_list = []

        for i in range(occupancy_kmeans_df.shape[0]):  # range(cluster number):
            dataset = occupancy_kmeans_df.iloc[i, :].values.astype('float16')
            dataset = dataset.reshape(-1, 1)
            
            # del occupancy_kmeans_df
            # gc.collect()
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            # reshape into X=t and Y=t+1
            look_back = 3  # the number of previous time steps to use as input variables to predict the next time period
            lookback_vector = create_dataset_fifteen(dataset, look_back)  # [x1,x2,x3,y]

            # split into train and test sets
            train, test=train_test_split(lookback_vector,test_size =0.3)
            trainX = np.array([i[0:3] for i in train])
            trainY = np.array([i[-1] for i in train])
            testX = np.array([i[0:3] for i in test])
            testY = np.array([i[-1] for i in test])

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(4, input_shape=(1, look_back)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            history = model.fit(trainX, trainY, validation_split=0.1, epochs=20, batch_size=1, verbose=2)

            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)

            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainY = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testY = scaler.inverse_transform([testY])

            # calculate root mean squared error
            trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
            train_score_list.append(trainScore)
            testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
            test_score_list.append(testScore)

        trainScore_Avg = averagenum(train_score_list)
        testScore_Avg = averagenum(test_score_list)
        print('Train Score Avg.: %.6f RMSE' % (trainScore_Avg))
        print('Test Score Avg.: %.6f RMSE' % (testScore_Avg))
        csvfile.write(str(nums) + ',' + str(trainScore_Avg) + ',' + str(testScore_Avg) + '\n')
        print('%f mins' % ((time.time() - start_time) / 60))
        print("-----------------------------------------------------------------------------")