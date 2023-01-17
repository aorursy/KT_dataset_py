import numpy as np

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt 
np.random.seed(1000)                                 # Generate Random  Numbers

X = np.linspace(-1,1,200)                            # Generate some input data X

np.random.shuffle(X)                                 # Shuffle the Input Data X

Y = 0.5 * X + 3.5 + np.random.normal(0,0.05,(200, )) # Generate Correlated Y

plt.scatter(X, Y)                                    # plot X and Y

plt.show()



X_train, Y_train = X[:150], Y[:150]                  # train test split

X_test , Y_test  = X[151:], Y[151:]       
model = Sequential() # Initialize Sequential Model

model.add(Dense(output_dim=1, input_dim=1)) 

model.compile(loss='mse', optimizer='sgd') # Compile the Model
model.fit(X_train, Y_train,batch_size=50, epochs=50)
cost = model.evaluate(X_test, Y_test, batch_size=10)

print('test cost:', cost)
Y_pred = model.predict(X_test)

plt.scatter(X_test, Y_test)

plt.plot(X_test, Y_pred)

plt.show()