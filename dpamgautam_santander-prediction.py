import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

submit = pd.read_csv("../input/sample_submission.csv")
x_train = train_data.iloc[:, 2:]

y_train = train_data.iloc[:, 1]

x_test = test_data.iloc[:, 1:]

output = submit.iloc[:, 0]

output = pd.DataFrame(output)
# method-1:



#x_train, y_train = np.array(x_train), np.array(y_train)



#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



#from sklearn.linear_model import LogisticRegression

#model1 = LogisticRegression()

#model1.fit(x_train, y_train)

#predictions1 = model1.score(x_test, y_test)

#print(predictions1)
# method-2:



x_train, y_train, x_test = np.array(x_train), np.array(y_train), np.array(x_test)



from sklearn.preprocessing import normalize, scale, StandardScaler

sc = StandardScaler()

#x_train = normalize(x_train)

#x_test = normalize(x_test)

#x_train = scale(x_train)

#x_test = scale(x_test)

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)



#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers



sgd = optimizers.SGD(lr=0.01)

rmsprop = optimizers.RMSprop(lr=0.01)

adagrad = optimizers.Adagrad(lr=0.01)

adadelta = optimizers.Adadelta(lr=0.01)

adam = optimizers.Adam(lr=0.001)

adamax = optimizers.Adamax(lr=0.01)



#x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)



model2 = Sequential()

model2.add(Dense(100, input_dim=200, kernel_initializer='normal', activation='relu'))

model2.add(Dense(80, kernel_initializer='normal', activation='relu'))

model2.add(Dense(60, kernel_initializer='normal', activation='relu'))

model2.add(Dense(40, kernel_initializer='normal', activation='relu'))

model2.add(Dense(20, kernel_initializer='normal', activation='relu'))

model2.add(Dense(10, kernel_initializer='normal', activation='relu'))

model2.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model2.fit(x_train, y_train, batch_size=32, epochs=1)

#model2.evaluate(x_test, y_test)



predictions2 = model2.predict(x_test)

predictions2 = np.where(predictions2 > 0.5, 1, 0)

predictions2 = pd.DataFrame(predictions2)
pred = pd.DataFrame(predictions2)

output["target"] = pred

output.to_csv("out.csv", index=False)