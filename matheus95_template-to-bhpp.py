!pip install tensorflow==1.15.4
!pip install tensorflow-gpu==1.15.4
import tensorflow.compat.v1 as tf
import tensorflow.contrib.layers as layers
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
import tensorflow.keras
import csv
%matplotlib inline
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
df.describe()
X_train, X_test, y_train, y_test = train_test_split(df [['RM', 'LSTAT', 'PTRATIO']], df[['target']], test_size=0.3, shuffle=True, random_state=0)
X_train = np.array(MinMaxScaler().fit_transform(X_train))   # Need to convert dataframe to np.array for use with Keras 
y_train = np.array(MinMaxScaler().fit_transform(y_train))

X_test =  np.array(MinMaxScaler().fit_transform(X_test))
y_test =  np.array(MinMaxScaler().fit_transform(y_test))
m = None
n = None   # Number of features
n_hidden = None  # Number of hidden neurons
def get_model():
    pass
model = get_model()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=5, batch_size=10, verbose=1)
y_test_pred = model.predict(X_test)
rmse = mean_squared_error( y_test, y_test_pred  )
print("RMSE = ", rmse)
def write_prediction_csv(prediction):
    with open('price_file.csv', mode='w') as price_file:
        price_writer = csv.writer(price_file, delimiter=',')
        price_writer.writerow(['Id', 'Expected'])
        for idx,value in enumerate(prediction):
            price_writer.writerow([f'{idx}', f'{value[0]}'])
all_base = np.vstack((X_test, X_train))
y_submit = model.predict(all_base)
y_submit = list(map(lambda x : x[0], y_submit))
write_prediction_csv(all_base)
best_epoch = None
best_batche = None
rmse_min = None

epochs = None
batches = None
for epoch in epochs:
    for batch in batches:
        model = get_model()
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=epoch, batch_size=batch, verbose=1)
        y_test_pred = model.predict(X_test)
        rmse = mean_squared_error( y_test, y_test_pred  )
        # Do a script to get better hyperparameters and set the values on best_epoch, best_batche, and rmse_min.
print("Minimum rmse = ", rmse_min)