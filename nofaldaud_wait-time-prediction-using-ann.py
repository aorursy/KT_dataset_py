#Importing necessary libaries

import pandas as pd

import numpy as np

import torch.optim as optim

from math import sqrt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import torch

import torch.nn as nn

import torch.nn.functional as fn
class Net(nn.Module):

    

    def __init__(self):

        super(Net, self).__init__()

        self.h1 = nn.Linear(14,28)

        

        self.h2 = nn.Linear(28,28)

        

        self.h3 = nn.Linear(28,28)

        

        self.o = nn.Linear(28,1)

        

    def forward(self,x):

            h1 = fn.relu(self.h1(x))

            h2 = fn.relu(self.h2(h1))    

            h3 = fn.relu(self.h3(h2))

            

            return self.o(h3)

        
df = pd.read_csv('../input/uio_clean.csv')



df= df.drop(['id','vendor_id','store_and_fwd_flag'], axis=1)

df.describe()

# removing outliers or erroneous values i.e. trip duration should be between 20 sec and 3hrs, distance should be

# between 100m and 100km, trip duration should be greater than wait time

df=df[(df['trip_duration'].between(30,7200)) & (df['dist_meters'].between(100,100000)) & (df['trip_duration']>df['wait_sec'])]

df=df[df['wait_sec'].between(0,7200)]

df=df[(df['pickup_longitude'].between(-80,-77)) & (df['pickup_latitude'].between(-4,1)) & (df['dropoff_longitude'].between(-80,-77))

      &(df['dropoff_latitude'].between(-4,1))]

df.shape[0]

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

df['hour_of_day']=df['pickup_datetime'].dt.hour

df['month'] = df['pickup_datetime'].dt.month

df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

df['day_of_year'] = df['pickup_datetime'].dt.dayofyear

df['week_of_year'] = df['pickup_datetime'].dt.weekofyear

df=df.drop(['pickup_datetime','dropoff_datetime'], axis=1)



df['trip_duration_log'] = np.round(np.log1p(df['trip_duration']), 5)

df['dist_meters_log'] = np.round(np.log1p(df['dist_meters']), 5)

df['avg_speed'] = df['dist_meters'] / df['trip_duration'] 

# avg speed should be between 3m/s and 30m/s or 108km/hr

df = df[df['avg_speed'].between(3,30)]



df=df.dropna()







#split dependent and independent variable

X = df.drop(['wait_sec'],axis=1)

#Taking natural log of the target variable, this helps the model converge better and gives better results

y = np.log1p(df['wait_sec'])



#Normalization function



Xscaler = StandardScaler()



# Spliting the data into training set and testing set in a 80/20 ratio



X_train, X_test, y_train, y_test = train_test_split(Xscaler.fit_transform(X), y, test_size=0.20, random_state=42)

# Concatenating X_train and y_train for the mini-batches for training the NN

Z = np.column_stack((X_train,y_train))

#A = np.array(A.values,dtype=np.float32)

Z.shape


def training(Z):

        learning_rate = 0.001

        batch_size = 100

        

        model = Net()

        

        #defining the loss function

        criterion = nn.SmoothL1Loss()

        

        #defining the optimizer

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)



        #Loop that defines the number of times the data is passed through the network



        for i in range(70):

            #shuffle the data every time

            np.random.shuffle(Z)

            start = 0



            while start < len(Z):

                #Zero the gradient buffers of all parameters and backprops with random gradients

                # clear previous gradients

                optimizer.zero_grad()

                end = start + batch_size if (start+batch_size) < len(Z) else len(Z)

                batch = Z[start:end,:]

                X = np.array(batch[:,:14],dtype=np.float32)

                y = np.array(batch[:,14],dtype=np.float32)

                start = end



                #Create a tensor from a numpy.ndarray

                X = torch.from_numpy(X)

                y = torch.from_numpy(y)

                y = y.unsqueeze(1)

                

                # Pass the input data through the Net using forward function

                p = model.forward(X)

                # Compute loss

                loss = criterion(p,y) 

                # compute gradients of all variables wrt loss

                # Propagate gradients back into the network’s parameters

                loss.backward()

                # Update the weights of the network

                # perform updates using calculated gradients

                optimizer.step()

            print(loss.item())

                

        return model
y_test = np.array(y_test.values,dtype=np.float32)

y_test.shape
def test(X_test,y_test,model):

        y_pred = np.zeros(len(y_test))

        for idx, x in enumerate(X_test):

                X = np.array(x,dtype=np.float32)

                

                X = torch.from_numpy(X)



                X = X.unsqueeze(0)

                p = model.forward(X)

                y_pred[idx] = p.item()

                

        mae_error = np.abs(y_test - y_pred)

        return y_pred,mae_error.mean()
# Function call to Train the model

model = training(Z)

# Function call to test the accuracy on test set

y_pred,mae_error=test(X_test,y_test,model)
print('MAE on test data is',mae_error)
y_test_act = np.expm1(y_test)

y_pred_act = np.expm1(y_pred)

mae = np.abs(y_test_act - y_pred_act)

print("The actual mean absolute test error is: ",mae.mean())