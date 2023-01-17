import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, BatchNormalization, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import os
#reading the files
data_dir = "../input/"
merchants = pd.read_csv(os.path.join(data_dir, "merchants.csv"))
historical = pd.read_csv(os.path.join(data_dir, "historical_transactions.csv"), )
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"),  parse_dates=["first_active_month"])
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"),  parse_dates=["first_active_month"])
new_trans_df = pd.read_csv(os.path.join(data_dir, "new_merchant_transactions.csv"))

gdf = historical.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_merch_transactions"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")
gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
train_df = pd.merge(train_df, gdf, on="card_id", how="left")
test_df = pd.merge(test_df, gdf, on="card_id", how="left")

train_df["year"] = train_df["first_active_month"].dt.year
test_df["year"] = test_df["first_active_month"].dt.year
train_df["month"] = train_df["first_active_month"].dt.month
test_df["month"] = test_df["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", 
               "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
               "min_hist_trans", "max_hist_trans",
               "year", "month","num_merch_transactions", 
                "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
                "min_merch_trans", "max_merch_trans",
              ]

#get train and test dataframes
train_X = train_df[cols_to_use]
test_X = test_df[cols_to_use]
train_y = train_df['target'].values
#scale the data and impute the null values 
#note: apparently, GPU environment doesn't have an updated version of sklearn,
#so we cannot use sklearn.impute.SimpleImputer. In CPU environement this is possible
sc = StandardScaler()
train_X = train_X.fillna(0)
train_X = sc.fit_transform(train_X)
x_train, x_val, y_train, y_val = train_test_split(train_X, train_y, test_size = .1, random_state = 42)
#building the network
import keras.backend as K
#definind the rmse metric
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

fh_neurons = 1024 #first hidden layer
drop_rate = 0.7

#the model is just a sequence of fully connected layers, batch normalization and dropout using ELUs as activation functions
model = Sequential()
model.add(Dense(fh_neurons, input_dim=x_train.shape[1], activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
model.add(Dense(fh_neurons*2, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
model.add(Dense(fh_neurons*2, activation='elu'))
model.add(Dense(fh_neurons, activation='elu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam',loss=rmse)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
checkpointer = ModelCheckpoint(filepath='weights.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=15, batch_size=256, callbacks = [early_stopping, checkpointer])
import matplotlib.pyplot as plt
#plotting training and validations losses
plt.plot(model.history.history['val_loss'], label = "val_loss")
plt.plot(model.history.history['loss'], label = "loss")
plt.legend()
#saving the card_ids
ids = test_df['card_id'].values
submission = pd.DataFrame(ids, columns=['card_id'])
#making the predictions
test_df = test_df[cols_to_use]
test_df = test_df.fillna(0)
test_df = sc.transform(test_df)
predictions = model.predict(test_df)
submission['target'] = predictions.flatten()
submission.head()
submission.to_csv("submission_neuralnet.csv", index = False, header = True)