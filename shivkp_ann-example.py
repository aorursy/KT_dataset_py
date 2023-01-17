import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# dataset link : https://raw.githubusercontent.com/anshulmahajan01/DataSets-for-practice/master/Car_sales.csv
dataset = pd.read_csv("../input/car-salescsv/Car_sales.csv")
dataset.head()
sns.pairplot(dataset)
dataset = dataset.drop(['Manufacturer','Model', 'Latest Launch'], axis = 1)
for column in dataset.columns:
    dataset[column] = pd.to_numeric(dataset[column],errors='coerce')

dataset.info()

X = dataset.iloc[:,4:].values
Y = dataset.iloc[:,3].values

# changing dimension of Y into 1D to 2D
Y = Y.reshape(-1,1)
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X)
X = imputer.transform(X)

imputer_y = imputer.fit(Y)
Y = imputer_y.transform(Y)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()

X = scale.fit_transform(X)

scale.data_max_
scale.data_min_
print(X)
Y = scale.fit_transform(Y)
print(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(40,input_dim=9, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
# for calculate r2 value
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_square])
epochs_hist = model.fit(X_train, Y_train, epochs=40, batch_size=40,  verbose=1, validation_split=0.2)
print(epochs_hist.history.keys())
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
# Engine_size, Horsepower, Wheelbase, Width, Length, Curb_weight, Fuel_capacity, Fuel_efficiency ,Power_perf_factor
# normalize data
X_test_sample = np.array([[0.2857, 0.29113,  0.26899, 0.42774, 0.407, 0.518, 0.21659, 0.3333, 0.344055]])
#Actual data
#X_test_sample = np.array([[3, 170,  105, 70, 180, 3.8, 15, 25, 80]])

y_predict_sample = model.predict(X_test_sample)
print('Expected Car Price in normalize form =', y_predict_sample)
y_predict_sample_orig = scale.inverse_transform(y_predict_sample)
print('Expected Car Price in thousands =', y_predict_sample_orig)