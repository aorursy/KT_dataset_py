import pandas as pd

import math

import numpy as np

import datetime



import matplotlib.pyplot as plt



import time



from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn import svm

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



from sklearn.neural_network import MLPClassifier



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM



from sklearn.metrics import r2_score



# from genetic_selection import GeneticSelectionCV   

# I have phone verification issue. So i can't import this module
def readDataFromCSV(file_path):

    data = pd.read_csv(file_path)

    data.reset_index()

    data = data.set_index("Date")

    data.index = pd.to_datetime(data.index)

    return data
def readDataAndWrite(stock_code, file_path):

    ticker = yf.Ticker(stock_code)

    data= ticker.history(period= "max")

    data.to_csv(r""+ file_path)
mydata = readDataFromCSV("../input/amazon-stock-price-prediction/mydata.csv")
mydata.columns
mydata
amazon_trends = readDataFromCSV("../input/amazon-stock-price-prediction/amazon_trends.csv")

mydata.loc[mydata.index & amazon_trends.index, "Google_Trends"] = amazon_trends["Google_Trends"]
mydata.columns
tomorrows_price = mydata["Close"].shift(-1)

tomorrows_price = tomorrows_price[0: -1]

mydata = mydata[0:-1]
price_movement = mydata["Close"] < tomorrows_price
min_max_scaler = preprocessing.MinMaxScaler()
norm_mydata = min_max_scaler.fit_transform(mydata)

norm_tomorrows_price = min_max_scaler.fit_transform(tomorrows_price[:, None])
norm_train_data, norm_test_data, norm_train_target, norm_test_target = train_test_split(norm_mydata, norm_tomorrows_price, test_size = 0.2, random_state = 0, shuffle = False)



norm_train_target_movement = norm_train_target < pd.DataFrame(norm_train_target).shift(-1)

norm_test_target_movement = norm_test_target < pd.DataFrame(norm_test_target).shift(-1)
k= int(len(price_movement)* 0.8)

train_price_movement = price_movement[0:k]

test_price_movement = price_movement[k:]
linear_reg = linear_model.LinearRegression()
linear_reg.fit(norm_train_data, norm_train_target)
close_column_index = 3
linear_reg_results = pd.DataFrame()

tmp = linear_reg.predict(norm_test_data)

linear_reg_results["Price_Predictions"] = tmp.reshape(len(tmp))

linear_reg_results["Price_Movement_Predictions"] = linear_reg_results["Price_Predictions"] < linear_reg_results["Price_Predictions"].shift(-1)
plt.figure(figsize = (7, 4))

plt.plot(norm_test_target[0:100], "red", label= 'Real')   ## Test Target

plt.plot(linear_reg_results["Price_Predictions"][0:100], "blue", label= 'Predicted')  ## Predicted



plt.title('Stock')

plt.xlabel('Time [days]')

plt.ylabel('Normalized Price')

plt.legend(loc= 'best')

plt.show()
print("R2 Metric: ", r2_score(norm_test_target, linear_reg_results["Price_Predictions"]))

print("Price Movement Accuracy: ", accuracy_score(linear_reg_results["Price_Movement_Predictions"], norm_test_target_movement))
plt.figure(figsize = (7, 4))

plt.title('Stock Movement Prediction')

plt.plot(linear_reg_results["Price_Movement_Predictions"])

plt.xlabel("Days")

plt.show()
linear_reg.fit(norm_train_data, train_price_movement)
linear_cls_results = pd.DataFrame()

tmp = linear_reg.predict(norm_test_data)

linear_cls_results["Price_Movement_Predictions"] = tmp > 0.5
print("Price Movement Accuracy: ", accuracy_score(linear_cls_results["Price_Movement_Predictions"], norm_test_target_movement))
plt.figure(figsize = (7, 4))

plt.title('Stock Movement Prediction')

plt.plot(linear_cls_results)

plt.xlabel("Days")

plt.show()
sv_regression = svm.SVR(C=10, degree = 2, kernel= 'sigmoid', tol= 0.1) 

sv_regression.fit(norm_train_data, norm_train_target)
svr_results = pd.DataFrame()

tmp = sv_regression.predict(norm_test_data)

svr_results["Price_Predictions"] = tmp.reshape(len(tmp))

svr_results["Price_Movement_Predictions"] = svr_results["Price_Predictions"] < svr_results["Price_Predictions"].shift(-1)
plt.figure(figsize= (7,4))

plt.plot(pd.DataFrame(norm_test_data)[close_column_index][0:10], "purple", label = "Data" )

plt.plot(svr_results["Price_Predictions"][0:10], "blue", label = "Predicted")

plt.plot(norm_test_target[0:10], "red", label = "Real")



plt.title('Stock (10 Days)')

plt.xlabel('Time [days]')

plt.ylabel('Normalized Price')

plt.legend(loc= 'best')

plt.show()



plt.show()
plt.figure(figsize= (7, 4))

plt.plot(norm_test_target, "red", label = "Real")

plt.plot(svr_results["Price_Predictions"], "blue", label= "Predicted")



plt.title('Stock')

plt.xlabel('Time [days]')

plt.ylabel('Normalized Price')

plt.legend(loc= 'best')

plt.show()
plt.figure(figsize = (7, 4))

plt.plot(svr_results["Price_Movement_Predictions"])

plt.title("Stock Movement Prediction")

plt.xlabel("Days")

plt.show()
print("R2 Metric: ", r2_score(norm_test_target, svr_results["Price_Predictions"]))

print("Price Movement Accuracy: ", accuracy_score(svr_results["Price_Movement_Predictions"], norm_test_target_movement))
sv_classification = svm.SVC(C= 10, gamma = 10, kernel = "rbf")

sv_classification.fit(norm_train_data, train_price_movement)
svc_results = pd.DataFrame()

tmp = sv_classification.predict(norm_test_data)

svc_results["Price_Movement_Predictions"] = tmp.reshape(len(tmp)) > 0.5
print("Price Movement Accuracy: ", accuracy_score(svc_results["Price_Movement_Predictions"], norm_test_target_movement))
plt.figure(figsize = (7, 4))

plt.plot(svc_results["Price_Movement_Predictions"])

plt.title("Stock Movement Prediction")

plt.xlabel("Days")

plt.show()
# svc_selector = GeneticSelectionCV(sv_classification,

#                               cv=5,

#                               verbose=1,

#                               scoring="accuracy",

#                               max_features=33,

#                               n_population=50,

#                               crossover_proba=0.5,

#                               mutation_proba=0.2,

#                               n_generations=40,

#                               crossover_independent_proba=0.5,

#                               mutation_independent_proba=0.05,

#                               tournament_size=7,

#                               n_gen_no_change=10,

#                               caching=True,

#                               n_jobs=-1)

# svc_selector = svc_selector.fit(norm_train_data, train_price_movement)
# accuracy_score(svc_selector.predict(norm_test_data), test_price_movement)
# plt.plot(svc_selector.predict(norm_test_data))
# accuracy_score(svc_selector.predict(norm_train_data), train_price_movement)
def createSequentialModel(input_size, activation_function):

    model = Sequential()



    model.add(LSTM(

        input_dim= input_size,

        units=50,

        return_sequences=True))

    model.add(Dropout(0.2))

    

    model.add(LSTM(

        100,

        return_sequences=False))

    model.add(Dropout(0.2))



    model.add(Dense(

        units=1, activation = activation_function))



    model.compile(loss='mse', optimizer='rmsprop')

    

    return model
def executeSequentialModel(model, data, target, test_split, epoch):

    min_max_scaler = preprocessing.MinMaxScaler()

    

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = test_split, random_state = 4, shuffle = False)



    fitted_train_data = min_max_scaler.fit_transform(train_data)

    fitted_test_data = min_max_scaler.fit_transform(test_data)





    train_target = list(train_target)

    test_target = list(test_target)

    

    lstm_train_data = np.reshape(fitted_train_data, (fitted_train_data.shape[0], 1, fitted_train_data.shape[1]))

    lstm_test_data = np.reshape(fitted_test_data, (fitted_test_data.shape[0],1, fitted_test_data.shape[1]))

    

    print(lstm_train_data.shape)

    print(lstm_test_data.shape)



    lstm_train_target = np.reshape(train_target, (len(train_target), 1))

    lstm_test_target = np.reshape(test_target, (len(test_target), 1))

    

    

    model.fit(

    lstm_train_data,

    lstm_train_target,

    batch_size=256,

    epochs= epoch,

    validation_split=0.2, shuffle = False)

    

    predictions = min_max_scaler.fit_transform(model.predict(lstm_test_data))

    

    predictions = predictions.reshape(len(predictions))

    

#     plt.plot(predictions)

#     plt.show()

    

    return accuracy_score(predictions > 0.5 , test_target, normalize = True), predictions
lstm_model = createSequentialModel(norm_train_data.shape[1], "hard_sigmoid")
lstm_train_data= norm_train_data.reshape(norm_train_data.shape[0],1, norm_train_data.shape[1])

lstm_test_data= norm_test_data.reshape(norm_test_data.shape[0],1, norm_test_data.shape[1])
history= lstm_model.fit(

    lstm_train_data,

    norm_train_target,

    batch_size=256,

    epochs= 300,

    validation_split=0.2, shuffle = False)
lstm_reg_results = pd.DataFrame()

tmp = lstm_model.predict(lstm_test_data)

lstm_reg_results["Price_Predictions"] = tmp.reshape(len(tmp))

lstm_reg_results["Price_Movement_Predictions"] = lstm_reg_results["Price_Predictions"] < lstm_reg_results["Price_Predictions"].shift(-1)
plt.figure(figsize = (7,4))

plt.plot(lstm_reg_results["Price_Predictions"], "blue", label = "Predicted")

plt.plot(norm_test_target,"red", label = "Real")



plt.title('Stock')

plt.xlabel('Time [days]')

plt.ylabel('Normalized Price')

plt.legend(loc= 'best')



plt.show()
plt.figure(figsize= (7,4))

plt.plot(history.history["val_loss"], "orange")

plt.title("Validation Loss")



plt.xlabel('Epoch')

plt.ylabel('Loss')



plt.show()
plt.figure(figsize= (7,4))

plt.plot(lstm_reg_results["Price_Movement_Predictions"])

plt.title("Stock Movement Prediction")

plt.xlabel("Days")

plt.show()
print("R2 Metric: ", r2_score(norm_test_target, lstm_reg_results["Price_Predictions"]))

print("Price Movement Accuracy: ", accuracy_score(lstm_reg_results["Price_Movement_Predictions"], norm_test_target_movement))
lstm_cls_model = createSequentialModel(norm_train_data.shape[1], "hard_sigmoid")
train_price_movement.shape
history= lstm_cls_model.fit(

    lstm_train_data,

    np.asarray(train_price_movement),

    batch_size=256,

    epochs= 40,

    validation_split=0.2, shuffle = False)
lstm_cls_results = pd.DataFrame()

tmp = lstm_cls_model.predict(lstm_test_data)

lstm_cls_results["Price_Movement_Predictions"] = tmp.reshape(len(tmp)) > 0.5
print("Price Movement Accuracy: ", accuracy_score(lstm_cls_results["Price_Movement_Predictions"], norm_test_target_movement))
plt.figure(figsize= (7,4))

plt.plot(history.history["val_loss"], "orange")

plt.title("Validation Loss")

plt.xlabel("Days")

plt.show()
plt.figure(figsize = (7, 4))

plt.plot(lstm_cls_results["Price_Movement_Predictions"])

plt.title("Stock Movement Prediction")

plt.xlabel("Days")



plt.show()
max_accuracy = 0

column_names = mydata.columns

added_columns = list()

tmp_columns = list()

test_split = 0.2

epoch = 10

for i in column_names:

    tmp_columns = added_columns.copy()

    tmp_columns.append(i)

    model = createSequentialModel(len(tmp_columns), "hard_sigmoid")

    print("Columns: ", tmp_columns)

    accuracy, predictions = executeSequentialModel(model, mydata[tmp_columns], price_movement, test_split, epoch)

    print("Current Accuracy: ", accuracy)

    if accuracy > max_accuracy: 

        added_columns.append(i)

        print("Max Accuracy: ", max_accuracy)

        max_accuracy = accuracy

    

    else : print("Max Accuracy: ", max_accuracy)
print("Added Columns: ", added_columns)

print("Max Accuracy: ", max_accuracy)