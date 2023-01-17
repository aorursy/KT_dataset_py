import numpy



x = numpy.linspace(0, 5, 100).reshape(-1, 1)



numpy.random.seed(7)



noise = numpy.random.normal(0, 0.1, x.size).reshape(x.shape)



y = 1.5 * x + 2.7 + noise
%matplotlib inline

import matplotlib.pyplot as pyplot



fig1 = pyplot.figure()

axes1 = pyplot.axes(title='Vizualization of the data')

scatter1 = axes1.scatter(x, y)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)
fig2 = pyplot.figure()

axes2 = pyplot.axes(title='Split data')

scatter2_train = axes2.scatter(x_train, y_train, label='training data')

scatter2_test = axes2.scatter(x_test, y_test, label='test data')

legend2 = fig2.legend()
%%capture

from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

layer1 = Dense(units=1, input_dim=1, activation='linear')

model.add(layer1)
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=500, verbose=False)
yp = model.predict(x)

axes1.plot(x, yp, color='cyan', label='fit')

axes1.legend()

fig1
train_loss_and_metrics = model.evaluate(x_train, y_train)

test_loss_and_metrics = model.evaluate(x_test, y_test)
print(train_loss_and_metrics)

print(test_loss_and_metrics)
print(layer1.get_weights())