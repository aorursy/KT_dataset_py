import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
subm=pd.read_csv('../input/sample_submission.csv')
train.diabetes.value_counts()
train.age.hist()
plt.show()
train.glucose_concentration.hist()
plt.show()
train.bmi.hist()
plt.show()
hidden_units=100
learning_rate=0.01
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100
model = Sequential()
model.add(Dense(hidden_units, input_dim=8, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(1, activation=output_layer_act))
sgd=optimizers.SGD(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])
train.head()
train_x=train.iloc[:,1:9]
train_x.head()
train_y=train.iloc[:,9]
train_y.head()
model.fit(train_x, train_y, epochs=no_epochs, batch_size=len(train),  verbose=2)
test_x=test.iloc[:,1:]
predictions = model.predict(test_x)
predictions
rounded = [int(round(x[0])) for x in predictions]
print(rounded)
subm.diabetes=rounded
subm.to_csv('submission.csv',index=False)
