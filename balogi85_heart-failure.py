import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = '../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(path)
data.head()
data.tail()
data.drop('time', axis=1, inplace=True)
data.describe()
data.shape
#sns.pairplot(data, hue="DEATH_EVENT", diag_kind="hist")
death = data['DEATH_EVENT'].value_counts()
death
death.plot.bar()
data.plot.scatter('platelets','age')
data.plot.scatter('creatinine_phosphokinase', 'age')
plt.scatter(data.ejection_fraction, data.age)
data.plot.scatter('serum_creatinine', 'age')
label = data['DEATH_EVENT']
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_train
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam, Nadam, RMSprop
model = Sequential()
model.add(Dense(500, activation='relu', input_dim = x_train.shape[1]))
model.add(Dense(200,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
s = SGD(learning_rate=0.0001)
a = Adam(learning_rate=0.01)
na = Nadam(learning_rate=0.0001)
rms = RMSprop(learning_rate=0.001)
model.compile(optimizer=s, loss = 'binary_crossentropy', metrics='accuracy')
history = model.fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test)) 
pd.DataFrame(history.history).plot(figsize=(18, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()





