import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#read input data file
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.backend import sign
from keras import optimizers
# change format to dataframe and add column names
fpath='/kaggle/input/eye_state.csv'
df=pd.read_csv(fpath, header=0, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', 'Target'])
df.head()
# check missing data in file
df.isna().sum()
# normalize the data
scaler = StandardScaler() 
scaled_values = scaler.fit_transform(df) 
df.loc[:,:] = scaled_values
df.head()
# prepare df for target results
y=pd.DataFrame({'Target':df['Target']})
y['Target']=y['Target'].astype(int)
y.head()
# remove results from the input df
df = df.drop(columns=['Target'])
df.head()
# change input format from df to array
df=np.array(df)
y=np.array(y)

# split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size=0.3)
print("Input shape ", df.shape)
print("Train data shapes ", X_train.shape,", ", Y_train.shape) 
print("Test data shapes ", X_test.shape,", ", Y_test.shape)
print ("X_train: ", X_train)
print ("Y_train: ", Y_train)
print("X_test: ", X_test)
print ("Y_test: ", Y_test)
# change input dimensions to 3 for LSTM input
X_train = np.reshape(X_train, (10485, 14, 1))
X_test = np.reshape(X_test, (4494, 14, 1))
# LSTM model
model = Sequential()

model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.15))
model.add(Dense(1, activation="sigmoid"))

model.summary()

adam_modified = optimizers.Adam(learning_rate=0.005, beta_1=0.7, beta_2=0.9, amsgrad=False)

model.compile(loss="binary_crossentropy", optimizer=adam_modified, metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=80)
def plot_predictions(test, predicted, title):
    plt.figure(figsize=(20,1))
    plt.plot(test, color='blue',label='Actual eye state')
    plt.plot(predicted, color='orange',label='Predicted eye state')
    # plt.title(title)
    plt.xlabel('Sample number')
    plt.ylabel('Eye state')
    # plt.legend()
    plt.show()
predictions = model.predict_classes(X_test)
Y_test=Y_test.astype('int32')
plot_predictions(Y_test[0:300], predictions[0:300], "Predictions made by LSTM model")
accuracy_score(Y_test, predictions)
predictions = predictions[:,0]
Y_test = Y_test[:,0]
X_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(X_tf)
    lossFunction = mean_squared_error(X_tf, predictions)
gradientFunction = tape.gradient(lossFunction, X_tf)
signed_grad = tf.sign(gradientFunction)
perturbed_data = X_test + 0.2 * signed_grad
X_perturbed = perturbed_data.numpy()
adversarial_predictions = model.predict_classes(X_perturbed)
accuracy_score(Y_test, adversarial_predictions)
x_test_plot = X_test[:, :, 0]
x_perturbed_plot = X_perturbed[:, :, 0]
plt.figure(figsize=(20,2))
plt.plot(x_test_plot[:50], color='blue',label='Actual data')
plt.plot(x_perturbed_plot[:50], color='orange',label='Perturbated data')
plt.xlabel('Sample number')
plt.ylabel('Eye state')
plt.show()
adversarial_predictions = model.predict_classes(X_perturbed)
plot_predictions(Y_test[0:300], adversarial_predictions[0:300], "Predictions made by LSTM model")
accuracy_score(Y_test, adversarial_predictions)