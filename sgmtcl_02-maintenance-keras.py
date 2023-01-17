import pandas as pd               

import numpy as np



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD
df_init = df = pd.read_csv('../input/maintenance_data.csv')



msk = np.random.rand(len(df)) < 0.8



#Convert strings to discreat integers

try:

    df.replace('TeamA',1, inplace=True)

    df.replace('TeamB',2, inplace=True)

    df.replace('TeamC',3, inplace=True)

    df.replace('Provider1',1, inplace=True)

    df.replace('Provider2',2, inplace=True)

    df.replace('Provider3',3, inplace=True)

    df.replace('Provider4',4, inplace=True)

except:

    pass  

Train = df[msk]

Test = df[~msk]



X_train = Train.drop('broken', axis=1).values

Y_train = Train.broken.values[np.newaxis].T



X_test = Test.drop('broken', axis=1).values

Y_test = Test.broken.values[np.newaxis].T



print(X_train.shape)

print(X_test.shape)
model = Sequential()



model.add(Dense(64, activation='relu', input_dim=6))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100)

print("Finished!")
score = model.evaluate(X_test, Y_test)

print(model.metrics_names)

print(score)