# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing Necessary Libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential, load_model

from keras.layers import LSTM, Dense, Dropout

from keras import optimizers

from keras.wrappers.scikit_learn import KerasClassifier

import os

import warnings

warnings.filterwarnings('ignore')
# Reading Data

rawdata = pd.read_csv(r"../input/apple-share-price-csv/Apple_Train.csv")

rawdata.head()
## All Columns available in excel sheet



# We have 11 columns, for price prediction i'm using only 5 columns

# All_Columns ["open","high","low","close","volume","adj_close", "year", "week_day", "week_no", "date", "month"]



# Using Columns For Prediction

using_columns = ["open","high","low","close","adj_close"]

using_columns





# Steps - How many steps model needs to look for predicting the values, or udating the weights

# Epochs - One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. 

    #An epoch is comprised of one or more batches.

Steps = 75

Epochs = 200
# Output columns/ Predective Column / Dependent columns - Index no starts from "0"

# We will be predecting "Close"(cloing prices) of the sahares

# We are taking "Close" as our Output columns

target_column_index = 3

target_column_index
## Creating the copy of data into new data frame, with only required 5 columns

df = rawdata[using_columns].copy()

df.head()
# Converting Un-Supervised Model To Supervised Model

# Defining Function For Creating Data Sets For LSTM

def build_timeseries(mat, y_col_index,TIME_STEPS):

    """

    Converts ndarray into timeseries format and supervised data format. 

    Takes first TIME_STEPS number of rows as input and sets the TIME_STEPS+1th 

    data as corresponding output and so on.

    :param mat: ndarray which holds the dataset

    :param y_col_index: index of column which acts as output

    :return: returns two ndarrays-- input and output in format suitable to feed

    to LSTM.

    """

    import pandas as pd

    import numpy as np

    from tqdm._tqdm_notebook import tqdm_notebook



    

    # total number of time-series samples would be len(mat) - TIME_STEPS

    dim_0 = mat.shape[0] - TIME_STEPS

    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    y = np.zeros((dim_0,))

    print("dim_0",dim_0)

    for i in tqdm_notebook(range(dim_0)):

        x[i] = mat[i:TIME_STEPS+i]

        y[i] = mat[TIME_STEPS+i, y_col_index]

#         if i < 10:

#           print(i,"-->", x[i,-1,:], y[i])

    print("length of time-series i/o",x.shape,y.shape)

    return x, y



# Help Taken From Tutorial - 

    # Link - https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944
# Spliting Data Into Train & Test - 

df_train, df_test = train_test_split(df, train_size=0.85, test_size=0.15, shuffle=False)

print("Train--Test size", len(df_train), len(df_test))



del df # Deleting Previous Data Frames
# Scaling the feature MinMax, build array

min_max_scaler = MinMaxScaler()

train_cols = using_columns

x = df_train.loc[:,train_cols].values

X_Train = min_max_scaler.fit_transform(x)

X_Test = min_max_scaler.transform(df_test.loc[:,train_cols])
# Creating Training Data - 

# Using Build_Timeseries Function for converting un-supervised data to supervised data



x_train, y_train = build_timeseries(X_Train,target_column_index, Steps)
# Model Creation Option 1

# Building a model - With Input Layer (100 Neurons),1 Dropout Layer, 1 Hidden Layer(100 Neurons) & Output Layer (1 Neuron) - 100/100/1

model = Sequential()

model.add(LSTM(units=100, return_sequences=True, 

               input_shape=(x_train.shape[1], len(using_columns))))

model.add(Dropout(0.005))

model.add(LSTM(units=100))

model.add(Dense(units=1))



# Compiling the function, Using Means Square Error as loss function, and learning rate of 0.00010

model.compile(loss='mean_squared_error', optimizer= optimizers.RMSprop(lr = 0.00010 ))



# Summary of model

model.summary()
# Implementing The Model

History = model.fit(x_train, y_train, epochs=Epochs, batch_size=32

                    ,verbose = 2, shuffle = True, validation_split = 0.05)

# model.save('stock_prediction.h5')
# Plot the results of the training.

plt.subplots(figsize=(9,5))

plt.plot(History.history['loss'], label="Training loss")

plt.plot(History.history['val_loss'], label="Validation loss")

plt.legend()

plt.show()
# Creating Test Data Set 

x_test, y_test = build_timeseries(X_Test,target_column_index, Steps)
# Making Prediction

predictions = model.predict(x_test)
# Calculating mean square error - For scaled values

MSE_Log = mean_squared_error(y_test, predictions)

print("Means Squared Error Is ",MSE_Log) 

del MSE_Log

MAE_Log = mean_absolute_error(y_test, predictions)

print("Means Absolute Error Is ", MAE_Log) 

del MAE_Log
# Inversing Data in actual values - 

# Total numbers of pridicted values

lenth = len(predictions)
# Working For inverse tranform the data (y_test)

Array_First = np.zeros((lenth,3)).astype(float)

Array_Last = np.zeros((lenth,1)).astype(float)

Array_Data = np.column_stack((Array_First, y_test, Array_Last))



y_test_inversed = np.array(min_max_scaler.inverse_transform(Array_Data)).astype(float)

y_test_inversed = y_test_inversed[:, target_column_index]

del Array_First, Array_Last, Array_Data





# Working For inverse tranform the data (predictions)

Array_First = np.zeros((lenth,3)).astype(float)

Array_Last = np.zeros((lenth,1)).astype(float)

Array_Data = np.column_stack((Array_First, predictions, Array_Last))



prediction_inversed = np.array(min_max_scaler.inverse_transform(Array_Data)).astype(float)

prediction_inversed = prediction_inversed[:, target_column_index]

del Array_First, Array_Last, Array_Data, lenth
# Calculating mean square error - For actual share price values

MSE_Normal = mean_squared_error(y_test_inversed, prediction_inversed)

print("Means Squared Error Is ",MSE_Normal) 

del MSE_Normal

MAE_Normal  = mean_absolute_error(y_test_inversed, prediction_inversed)

print("Means Absolute Error Is ", MAE_Normal ) 

del MAE_Normal

R2_Score_Normal = r2_score(y_test_inversed, prediction_inversed)

print("R2 Score Is ", R2_Score_Normal ) 

del R2_Score_Normal
# Ploting actual figures and predicted figures

plt.subplots(figsize=(19,6))

plt.plot(y_test_inversed, color='red',  label= "True Price")

plt.plot(prediction_inversed, color='blue',  label="Predicted Price")

plt.title("Steps = {} & Epoch = {}". format(Steps,Epochs ),

         color='red')

plt.legend()

plt.show()
# Creating New Data Frame - Showing Predicted Values, Original Values & Difference (Predicted Values Minus Original Values)

Result_Data = pd.DataFrame({"Original_Values" : y_test_inversed, "Predicted_Values" : prediction_inversed})

Result_Data["Pred_Minus_Original"] = Result_Data["Predicted_Values"] - Result_Data["Original_Values"]

Result_Data.head()
# Slicing Data (Dates And Close Value) From Original Date Set

start_point = len(rawdata) - len(y_test)

Dates = rawdata.iloc[start_point:,[0, 4]].reset_index(drop = True)

Dates.head(3)
# Concatenating Both The Data Frames, Result_Data & Dates - For Better visibility of our predicted close values and original close values

Result_Data = pd.concat([Dates['date_full'], Result_Data], axis=1)

Result_Data.set_index('date_full', inplace = True)

Result_Data.head()
# Ploting actual figures and predicted figures

plt.subplots(figsize=(19,6))



plt.plot(Result_Data.Predicted_Values, color='red',  label= "True Price")

plt.plot(Result_Data.Original_Values, color='blue',  label="Predicted Price")

plt.plot(Result_Data.Pred_Minus_Original, color='green',  label="Predicted Price")



plt.legend()

plt.show()