import numpy as np 

import pandas as pd 



import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style("whitegrid")

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import datetime

from time import mktime



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error



import keras

from keras.models import Sequential

from keras.layers import Dense, LSTM, GRU, Embedding, Dropout

from keras.optimizers import RMSprop

from keras import backend as K
data = pd.read_csv("../input/percolata/Percolata.csv")

print(data.shape)

data.head()
def preprocess_data(df, busind_id):

    new_df = df[df["BUSIND_ID"] == busind_id]

    new_df["TIME"] = new_df["FISCAL_DATE"] + " " + new_df["PERIOD_ID"]

    new_df["TIME"] = pd.to_datetime(new_df["TIME"], format="%Y-%m-%d %H:%M:%S")

    new_df = new_df.sort_values(by="TIME")

    new_df = new_df.reset_index(drop=True)

    return new_df





id_one = preprocess_data(data, 1)

id_two = preprocess_data(data, 2)

id_three = preprocess_data(data, 3)

id_four = preprocess_data(data, 4)

id_five = preprocess_data(data, 5)
def convert_unixtimes(times):

    datetimes = list(map(lambda time: time.to_pydatetime(), list(times)))

    unixtimes = list(map(lambda time: mktime(time.timetuple()), datetimes))

    return pd.Series(unixtimes)



def convert_datetime(unixtime):

    time = datetime.datetime.utcfromtimestamp(unixtime)

    date = time.strftime("%Y-%m-%d")

    period = time.strftime("%H:%M:%S")

    return date, period
id_one
def limit_max_min(lim, lim_max=True):

    if lim_max:

        return lambda x: x if x < lim else lim

    else:

        return lambda x: x if x > lim else lim



id_one["GROUND_TRUTH"] = id_one["GROUND_TRUTH"].apply(limit_max_min(15, lim_max=True))

id_two["GROUND_TRUTH"] = id_two["GROUND_TRUTH"].apply(limit_max_min(20, lim_max=True))

id_three["GROUND_TRUTH"] = id_three["GROUND_TRUTH"].apply(limit_max_min(500, lim_max=True))

id_three["GROUND_TRUTH"] = id_three["GROUND_TRUTH"].apply(limit_max_min(0, lim_max=False))

id_four["GROUND_TRUTH"] = id_four["GROUND_TRUTH"].apply(limit_max_min(120, lim_max=True))

id_four["GROUND_TRUTH"] = id_four["GROUND_TRUTH"].apply(limit_max_min(0, lim_max=False))

id_five["GROUND_TRUTH"] = id_five["GROUND_TRUTH"].apply(limit_max_min(40, lim_max=True))
id_one["GROUND_TRUTH"].max()
def plot_data(df, start, end, size=(30, 10)):

    df["TIME"] = df["FISCAL_DATE"] + " " + df["PERIOD_ID"]

    df["TIME"] = pd.to_datetime(df["TIME"], format="%Y-%m-%d %H:%M:%S")

    df.set_index("TIME")["GROUND_TRUTH"][start:end].plot(figsize=size)



plot_data(id_one, "2015-01-01", "2018-01-02")
id_one = [id_one.set_index("TIME"), id_two.set_index("TIME")]

id_two = [id_three.set_index("TIME"), id_four.set_index("TIME"), id_five.set_index("TIME")]



concated = pd.concat(id_one, axis=1)

concated_two = pd.concat(id_two, axis=1)

concated = concated.drop(["FISCAL_DATE", "PERIOD_ID", "BUSIND_ID"], axis=1)

concated_two = concated_two.drop(["FISCAL_DATE", "PERIOD_ID", "BUSIND_ID"], axis=1)

concated.columns = ["GROUND_TRUTH1", "GROUND_TRUTH2"]

concated_two.columns = ["GROUND_TRUTH3", "GROUND_TRUTH4", "GROUND_TRUTH5"]

concated.head()
concated_two.head()
print(pd.isnull(concated).sum()) 

print(pd.isnull(concated_two).sum()) 
copy = concated.copy()

copy.dropna(inplace = True)

sns.distplot(copy["GROUND_TRUTH1"])

sns.distplot(copy["GROUND_TRUTH2"])
concated["GROUND_TRUTH1"].fillna(0, inplace = True)

concated["GROUND_TRUTH2"].fillna(0, inplace = True)

print(pd.isnull(concated).sum()) 
def scale_data(data, column, feature_range=(0, 1)):

    scaler = MinMaxScaler(feature_range=feature_range)

    #scaler = StandardScaler()

    col = np.array(data[column]).reshape(-1, 1)

    scaler.fit(col)

    return scaler



scaler_one = scale_data(concated, "GROUND_TRUTH1")

scaler_two = scale_data(concated, "GROUND_TRUTH2")

scaler_three = scale_data(concated_two, "GROUND_TRUTH3")

scaler_four = scale_data(concated_two, "GROUND_TRUTH4")

scaler_five = scale_data(concated_two, "GROUND_TRUTH5")



concated["GROUND_TRUTH1"] = scaler_one.transform(np.array(concated["GROUND_TRUTH1"]).reshape(-1, 1))

concated["GROUND_TRUTH2"] = scaler_two.transform(np.array(concated["GROUND_TRUTH2"]).reshape(-1, 1))

concated_two["GROUND_TRUTH3"] = scaler_three.transform(np.array(concated_two["GROUND_TRUTH3"]).reshape(-1, 1))

concated_two["GROUND_TRUTH4"] = scaler_four.transform(np.array(concated_two["GROUND_TRUTH4"]).reshape(-1, 1))

concated_two["GROUND_TRUTH5"] = scaler_five.transform(np.array(concated_two["GROUND_TRUTH5"]).reshape(-1, 1))



concated.head()
shift = 0



def shift_column(data, column, shift_factor):

    new_table = data.copy()

    new_col = data[column].shift(shift_factor)

    new_table["SHIFTED"] = new_col

    new_table["SHIFTED"].fillna(0, inplace=True)

    return new_table



table_one = shift_column(concated, "GROUND_TRUTH1", shift)

table_two = shift_column(concated, "GROUND_TRUTH2", shift)



table_three = shift_column(concated_two, "GROUND_TRUTH3", shift)

table_four = shift_column(concated_two, "GROUND_TRUTH4", shift)

table_five = shift_column(concated_two, "GROUND_TRUTH5", shift)



table_one.head()
def train_valid_test_split(data, training_size, valid_size):

    train_df = data.iloc[:training_size]

    valid_df = data.iloc[training_size:training_size+valid_size]

    test_df = data.iloc[training_size+valid_size:]

    print("Training Set Size: " + str(train_df.shape))

    print("Validation Set Size: " + str(valid_df.shape))

    print("Testing Set Size: " + str(test_df.shape))

    return train_df, valid_df, test_df



training_size_one = 51277

training_size_two = 69321

validation_size = 10000



train_one, valid_one, test_one = train_valid_test_split(table_one, training_size_one, validation_size)

train_two, valid_two, test_two = train_valid_test_split(table_two, training_size_one, validation_size)

train_three, valid_three, test_three = train_valid_test_split(table_three, training_size_two, validation_size)

train_four, valid_four, test_four = train_valid_test_split(table_four, training_size_two, validation_size)

train_five, valid_five, test_five = train_valid_test_split(table_five, training_size_two, validation_size)
timestep = 3



def input_output(data, timesteps):

    dim0 = data.shape[0] - timesteps

    dim1 = data.shape[1] - 1 #no label column

    X = np.zeros((dim0, timesteps, dim1))

    y = np.zeros((dim0, ))

    

    for i in range(dim0):

        X[i] = data.iloc[i:timesteps+i, :-1]

        y[i] = data.iloc[timesteps+i, -1]

        

    return X, y



def trim_dataset(data, batch_size):

    rows_to_drop = data.shape[0]%batch_size

    if rows_to_drop:

        return data[:-rows_to_drop]

    else:

        return data

    

BATCH_SIZE = 1000



train_X_one, train_y_one = input_output(train_one, timestep)

train_X_one, train_y_one = trim_dataset(train_X_one, BATCH_SIZE), trim_dataset(train_y_one, BATCH_SIZE)

train_X_two, train_y_two = input_output(train_two, timestep)

train_X_two, train_y_two = trim_dataset(train_X_two, BATCH_SIZE), trim_dataset(train_y_two, BATCH_SIZE)

train_X_three, train_y_three = input_output(train_three, timestep)

train_X_three, train_y_three = trim_dataset(train_X_three, BATCH_SIZE), trim_dataset(train_y_three, BATCH_SIZE)

train_X_four, train_y_four = input_output(train_four, timestep)

train_X_four, train_y_four = trim_dataset(train_X_four, BATCH_SIZE), trim_dataset(train_y_four, BATCH_SIZE)

train_X_five, train_y_five = input_output(train_five, timestep)

train_X_five, train_y_five = trim_dataset(train_X_five, BATCH_SIZE), trim_dataset(train_y_five, BATCH_SIZE)

valid_X_one, valid_y_one = input_output(valid_one, timestep)

valid_X_one, valid_y_one = trim_dataset(valid_X_one, BATCH_SIZE), trim_dataset(valid_y_one, BATCH_SIZE)

valid_X_two, valid_y_two = input_output(valid_two, timestep)

valid_X_two, valid_y_two = trim_dataset(valid_X_two, BATCH_SIZE), trim_dataset(valid_y_two, BATCH_SIZE)

valid_X_three, valid_y_three = input_output(valid_three, timestep)

valid_X_three, valid_y_three = trim_dataset(valid_X_three, BATCH_SIZE), trim_dataset(valid_y_three, BATCH_SIZE)

valid_X_four, valid_y_four = input_output(valid_four, timestep)

valid_X_four, valid_y_four = trim_dataset(valid_X_four, BATCH_SIZE), trim_dataset(valid_y_four, BATCH_SIZE)

valid_X_five, valid_y_five = input_output(valid_five, timestep)

valid_X_five, valid_y_five = trim_dataset(valid_X_five, BATCH_SIZE), trim_dataset(valid_y_five, BATCH_SIZE)

test_X_one, test_y_one = input_output(test_one, timestep)

test_X_one, test_y_one = trim_dataset(test_X_one, BATCH_SIZE), trim_dataset(test_y_one, BATCH_SIZE)

test_X_two, test_y_two = input_output(test_two, timestep)

test_X_two, test_y_two = trim_dataset(test_X_two, BATCH_SIZE), trim_dataset(test_y_two, BATCH_SIZE)

test_X_three, test_y_three = input_output(test_three, timestep)

test_X_three, test_y_three = trim_dataset(test_X_three, BATCH_SIZE), trim_dataset(test_y_three, BATCH_SIZE)

test_X_four, test_y_four = input_output(test_four, timestep)

test_X_four, test_y_four = trim_dataset(test_X_four, BATCH_SIZE), trim_dataset(test_y_four, BATCH_SIZE)

test_X_five, test_y_five = input_output(test_five, timestep)

test_X_five, test_y_five = trim_dataset(test_X_five, BATCH_SIZE), trim_dataset(test_y_five, BATCH_SIZE)



"""

train_X_one.to_csv("train_X_one.csv")

train_y_one.to_csv("train_y_one.csv")

train_X_two.to_csv("train_X_two.csv")

train_y_two.to_csv("train_y_two.csv")

train_X_three.to_csv("train_X_three.csv")

train_y_three.to_csv("train_y_three.csv")

train_X_four.to_csv("train_X_four.csv")

train_y_four.to_csv("train_y_four.csv")

train_X_five.to_csv("train_X_five.csv")

train_y_five.to_csv("train_y_five.csv")

valid_X_one.to_csv("valid_X_one.csv")

valid_y_one.to_csv("valid_y_one.csv")

valid_X_two.to_csv("valid_X_two.csv")

valid_y_two.to_csv("valid_y_two.csv")

valid_X_three.to_csv("valid_X_three.csv")

valid_y_three.to_csv("valid_y_three.csv")

valid_X_four.to_csv("vaid_X_four.csv")

valid_y_four.to_csv("valid_y_four.csv")

valid_X_five.to_csv("valid_X_five.csv")

valid_y_five.to_csv("valid_y_five.csv")

test_X_one.to_csv("test_X_one.csv")

test_y_one.to_csv("test_y_one.csv")

test_X_two.to_csv("test_X_two.csv")

test_y_two.to_csv("test_y_two.csv")

test_X_three.to_csv("test_X_three.csv")

test_y_three.to_csv("test_y_three.csv")

test_X_four.to_csv("test_X_four.csv")

test_y_four.to_csv("test_y_four.csv")

test_X_five.to_csv("test_X_five.csv")

test_y_five.to_csv("test_y_five.csv")

"""
def lstm_model(data, target, valid_data, valid_target, nodes, epochs, batch_size):

    #define model

    model = Sequential()

    #add layers

    model.add(LSTM(nodes, batch_input_shape=(batch_size, data.shape[1], data.shape[2]), return_sequences=True, stateful=True))

    model.add(Dropout(0.3))

    model.add(LSTM(nodes//2))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    #compile model

    model.compile(optimizer="adam", loss="mae", metrics=["acc", "mae"])

    #model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc", "mae"])

    

    losses = []

    val_losses = []

    #fit model

    for i in range(epochs):

        print("Training Epoch: " + str(i+1) + " out of " + str(epochs) + " Epochs.")

        fit = model.fit(data, 

                  target,

                  nb_epoch=1,

                  batch_size=batch_size,

                  verbose=1,

                  shuffle=False,

                  validation_data=(valid_data, valid_target))

        losses.append(fit.history["loss"])

        val_losses.append(fit.history["val_loss"])

        model.reset_states()

        print("Resetting Model States")

    

    #visualize training and testing loss over epochs

    plt.figure(figsize=(15, 10))

    plt.plot(losses, label="training")

    plt.plot(val_losses, label="validation")

    plt.title("Training and Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(loc=7)

        

    return model
NODES = 128

EPOCHS = 10



model_one = lstm_model(train_X_one, train_y_one, valid_X_one, valid_y_one, NODES, EPOCHS, BATCH_SIZE)

model_two = lstm_model(train_X_two, train_y_two, valid_X_two, valid_y_two, NODES, EPOCHS, BATCH_SIZE)

model_three = lstm_model(train_X_three, train_y_three, valid_X_three, valid_y_three, NODES, EPOCHS, BATCH_SIZE)

model_four = lstm_model(train_X_four, train_y_four, valid_X_four, valid_y_four, NODES, EPOCHS, BATCH_SIZE)

model_five = lstm_model(train_X_five, train_y_five, valid_X_five, valid_y_five, NODES, EPOCHS, BATCH_SIZE)
model_one.summary()
def evaluate(model, test_X, test_y, scaler, batch_size):

    predictions = model.predict(test_X, batch_size=batch_size)

    table = pd.DataFrame({"Real": [], "Predictions": []})

    table["Real"] = pd.Series(test_y)

    table["Predictions"] = pd.Series(predictions.reshape((predictions.shape[0],)))

    table["Real"] = scaler.inverse_transform(np.array(table["Real"]).reshape(-1, 1))

    table["Predictions"] = scaler.inverse_transform(np.array(table["Predictions"]).reshape(-1, 1))

    return table



def rmse(real, predictions):

    return np.sqrt(mean_squared_error(real, predictions))

    

predictions_one = evaluate(model_one, test_X_one, test_y_one, scaler_one, BATCH_SIZE)

predictions_two = evaluate(model_two, test_X_two, test_y_two, scaler_two, BATCH_SIZE)

predictions_three = evaluate(model_three, test_X_three, test_y_three, scaler_three, BATCH_SIZE)

predictions_four = evaluate(model_four, test_X_four, test_y_four, scaler_four, BATCH_SIZE)

predictions_five = evaluate(model_five, test_X_five, test_y_five, scaler_five, BATCH_SIZE)



rmses = [rmse(predictions_one["Real"], predictions_one["Predictions"]), rmse(predictions_two["Real"], predictions_two["Predictions"]), rmse(predictions_three["Real"], 

         predictions_three["Predictions"]), rmse(predictions_four["Real"], predictions_four["Predictions"]), rmse(predictions_five["Real"], predictions_five["Predictions"])]

                                                                                                            

for i, rmse in enumerate(rmses):

    print("RMSE for ID {} is: ".format(i+1) + str(rmse))
predictions_one["Predictions"].min()
def plot_preds(lim, data):

    plt.figure(figsize=(15, 10))

    plt.plot(data.iloc[:lim].iloc[:, -1:])

    plt.plot(data.iloc[:lim].iloc[:, -2:-1])

    plt.xlabel("Time (15-min intervals)")

    plt.ylabel("Ground Truth")

    plt.title("Real vs Prediction")

    plt.legend(("Prediction", "Real"), loc=1)



plot1 = plot_preds(100, predictions_one)
def input_output2(data, timesteps):

    #dim0 = data.shape[0] 

    dim0 = 1

    dim1 = data.shape[1] 

    X = np.zeros((dim0, timesteps, dim1))

    #y = np.zeros((dim0, ))

    

    #X[i] = data.iloc[i:timesteps+i, :-1]

    X[0] = data.iloc[0:timesteps]

    #y[i] = data.iloc[timesteps+i, -1]

        

    return X
def datetime_hour(time):

    return datetime.datetime.strptime(time, "%H:%M:%S").hour
def predict_one(start_time, end_time, model_one, model_two, scaler_one, scaler_two, table, timesteps):

    pred_table_one = pd.DataFrame({"BUSIND_ID": [], "FISCAL_DATE": [], "PERIOD_ID": [], "GROUND_TRUTH": []})

    pred_table_two = pd.DataFrame({"BUSIND_ID": [], "FISCAL_DATE": [], "PERIOD_ID": [], "GROUND_TRUTH": []})

    table = table.iloc[-timesteps:]

    curr_inputs = input_output2(table, timesteps)

    while start_time <= end_time:

        curr_date, curr_period = convert_datetime(start_time)

        if datetime_hour(curr_period) < 7 or (curr_period != "22:00:00" and datetime_hour(curr_period) > 21):

            pred_table_one = pred_table_one.append({"BUSIND_ID": 1, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": 0}, ignore_index=True)

            pred_table_two = pred_table_two.append({"BUSIND_ID": 2, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": 0}, ignore_index=True)

            start_time += 900

            continue

        one_pred = model_one.predict_on_batch(curr_inputs)[0][0]

        two_pred = model_two.predict_on_batch(curr_inputs)[0][0]

        pred_table_one = pred_table_one.append({"BUSIND_ID": 1, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": one_pred}, ignore_index=True)

        pred_table_two = pred_table_two.append({"BUSIND_ID": 2, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": two_pred}, ignore_index=True)

        table = table.append({"GROUND_TRUTH1": one_pred, "GROUND_TRUTH2": two_pred}, ignore_index=True)

        table = table.iloc[-timesteps:]

        curr_inputs = input_output2(table, timesteps)

        start_time += 900

    pred_table_one["GROUND_TRUTH"] = scaler_one.inverse_transform(np.array(pred_table_one["GROUND_TRUTH"]).reshape(-1, 1))

    pred_table_two["GROUND_TRUTH"] = scaler_two.inverse_transform(np.array(pred_table_two["GROUND_TRUTH"]).reshape(-1, 1))

    return pred_table_one, pred_table_two
def predict_two(start_time, end_time, model_three, model_four, model_five, scaler_three, scaler_four, scaler_five, table, timesteps):

    pred_table_three = pd.DataFrame({"BUSIND_ID": [], "FISCAL_DATE": [], "PERIOD_ID": [], "GROUND_TRUTH": []})

    pred_table_four = pd.DataFrame({"BUSIND_ID": [], "FISCAL_DATE": [], "PERIOD_ID": [], "GROUND_TRUTH": []})

    pred_table_five = pd.DataFrame({"BUSIND_ID": [], "FISCAL_DATE": [], "PERIOD_ID": [], "GROUND_TRUTH": []})

    table = table.iloc[-timesteps:]

    curr_inputs = input_output2(table, timesteps)

    while start_time <= end_time:

        curr_date, curr_period = convert_datetime(start_time)

        three_pred = model_three.predict_on_batch(curr_inputs)[0][0]

        four_pred = model_four.predict_on_batch(curr_inputs)[0][0]

        five_pred = model_five.predict_on_batch(curr_inputs)[0][0]

        pred_table_three = pred_table_three.append({"BUSIND_ID": 3, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": three_pred}, ignore_index=True)

        pred_table_four = pred_table_four.append({"BUSIND_ID": 4, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": four_pred}, ignore_index=True)

        pred_table_five = pred_table_five.append({"BUSIND_ID": 5, "FISCAL_DATE": curr_date, "PERIOD_ID": curr_period, "GROUND_TRUTH": five_pred}, ignore_index=True)

        table = table.append({"GROUND_TRUTH3": three_pred, "GROUND_TRUTH4": four_pred, "GROUND_TRUTH5": five_pred}, ignore_index=True)

        table = table.iloc[-timesteps:]

        curr_inputs = input_output2(table, timesteps)

        start_time += 900

    pred_table_three["GROUND_TRUTH"] = scaler_three.inverse_transform(np.array(pred_table_three["GROUND_TRUTH"]).reshape(-1, 1))

    pred_table_four["GROUND_TRUTH"] = scaler_four.inverse_transform(np.array(pred_table_four["GROUND_TRUTH"]).reshape(-1, 1))

    pred_table_five["GROUND_TRUTH"] = scaler_five.inverse_transform(np.array(pred_table_five["GROUND_TRUTH"]).reshape(-1, 1))

    return pred_table_three, pred_table_four, pred_table_five
START_TIME = 1546300800

END_TIME = 1548979200 



preds_id_one, preds_id_two = predict_one(START_TIME, END_TIME, model_one, model_two, scaler_one, scaler_two, concated, timestep)

preds_id_one
preds_id_three, preds_id_four, preds_id_five = predict_two(START_TIME, END_TIME, model_three, model_four, model_five, scaler_three, scaler_four, scaler_five, concated_two, timestep)

preds_id_five
all_predictions = [preds_id_one, preds_id_two, preds_id_three, preds_id_three, preds_id_four, preds_id_five]

percolata_predictions = pd.concat(all_predictions)

percolata_predictions["GROUND_TRUTH"] = list(map(lambda x: round(x), percolata_predictions["GROUND_TRUTH"]))

percolata_predictions["BUSIND_ID"] = list(map(lambda x: int(x), percolata_predictions["BUSIND_ID"]))

percolata_predictions
percolata_predictions.to_csv("Percolata - January 2019 Predictions.csv")