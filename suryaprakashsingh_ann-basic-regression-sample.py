# Import libraries
import pandas as pd, numpy as np, tensorflow as tf
tf.__version__
dataset = pd.read_excel('../input/analysis-of-breaking-of-machineries/Combined_Cycle_powerplant.xlsx')
print(dataset.head(5))

X = dataset.iloc[:, :-1].values
# print(X)
y = dataset.iloc[:, -1].values
# print(y)
# Splitting the dataset in train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Initiliase the ANN
ann = tf.keras.models.Sequential()

# Adding the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second higgen layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))
# Compiling the ANN Model
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN Model
ann.fit(X_train, y_train, batch_size=32, epochs=100)
# Predicting the results
y_pred = ann.predict(X_test)

np.set_printoptions(precision=2)
print("Predicted vs Actual Values")
print(np.concatenate((np.array(y_pred).reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test), 1)), axis=1))