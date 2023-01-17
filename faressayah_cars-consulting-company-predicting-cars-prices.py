import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 80)
pd.set_option("display.float", "{:.2f}".format)
data = pd.read_csv('/kaggle/input/automobile-consulting-company/CarPrice_Assignment.csv')
data.head()
data.info()
data.describe()
for column in data.columns:
    print(f"{column}: {data[column].nunique()}")
    if data[column].nunique() < 10:
        print(f"{data[column].value_counts()}")
    print("====================================")
data.car_ID.nunique()
data.set_index('car_ID', inplace=True)
data.index.name = None
data.head()
sns.distplot(data.price, bins=30)
discrete_col = ['fueltype', 'fuelsystem', 'aspiration', 'doornumber', 'carbody', 
                'drivewheel', 'enginelocation', 'enginetype', 'symboling', 
                'cylindernumber']
plt.figure(figsize=(15, 15))
for i, column in enumerate(discrete_col, 1):
    plt.subplot(4, 3, i)
    sns.scatterplot(column, 'price', data=data, hue='price')
continous_col = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
                 'curbweight', 'enginesize', 'boreratio', 'stroke', 
                 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 
                 'compressionratio']
plt.figure(figsize=(15, 15))
for i, column in enumerate(continous_col, 1):
    plt.subplot(4, 4, i)
    sns.scatterplot(column, 'price', data=data, hue='price')
continous_col = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
                 'curbweight', 'enginesize', 'boreratio', 'stroke', 
                 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 
                 'compressionratio']
plt.figure(figsize=(15, 15))
for i, column in enumerate(continous_col, 1):
    plt.subplot(4, 4, i)
    sns.distplot(data[column])
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True)
data.CarName.nunique()
data['brand'] = data.CarName.str.split(' ', expand=True)[0].str.lower()

data['brand'] = data.brand.str.replace('toyouta', 'toyota')
data['brand'] = data.brand.str.replace('porcshce', 'porsche')
data['brand'] = data.brand.str.replace('maxda', 'mazda')
data['brand'] = data.brand.str.replace('vw', 'volkswagen').str.replace('vokswagen', 'volkswagen')
data['brand'] = data.brand.str.replace('vw', 'volkswagen').str.replace('vokswagen', 'volkswagen')

data.brand.value_counts()
plt.figure(figsize=(8, 5))
sns.scatterplot('brand', 'price', data=data, hue='price')
data.brand.nunique()
data.drop('CarName', axis=1, inplace=True)
object_features = data.select_dtypes(include=('object')).columns.tolist()
object_features
data_1 = pd.get_dummies(data, columns=object_features, drop_first=True, dtype='uint8')
data_1.dtypes
data_1.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data_1.drop('price', axis=1)
y = data_1.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
from sklearn import metrics

def print_evaluate(true, predicted, train=True):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    if train:
        print("========Training Result=======")
        print(f'MAE:\t\t {mae:.4f}')
        print(f'MSE:\t\t {mse:.4f}')
        print(f'RMSE:\t\t {rmse:.4f}')
        print(f'R2 Square:\t {r2_square:.4f}')
    elif not train:
        print("=========Testing Result=======")
        print(f'MAE:\t\t {mae:.4f}')
        print(f'MSE:\t\t {mse:.4f}')
        print(f'RMSE:\t\t {rmse:.4f}')
        print(f'R2 Square:\t {r2_square:.4f}')
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)
from sklearn.feature_selection import RFE

rfe_30 = RFE(lr_model, 30)
rfe_30.fit(X_train, y_train)

print(rfe_30.support_)
print(rfe_30.ranking_)
y_train_pred = rfe_30.predict(X_train)
y_test_pred = rfe_30.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(0.01), loss='mse')
r = model.fit(X_train, y_train,
              validation_data=(X_test,y_test),
              batch_size=512,
              epochs=200)
plt.figure(figsize=(10, 6))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)