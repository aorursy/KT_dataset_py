from numpy import array

from keras.models import Sequential

from keras.layers import LSTM # long short time memory

from keras.layers import Dense

from matplotlib import pyplot as plt


#LOC - Line of Code

data = [20, 40, 50, 70, 90, 120, 130, 160, 180]



plt.plot(data)

plt.show()
def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)
n_steps = 3

X, y = split_sequence(data, n_steps)

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))#adding a layer

model.add(Dense(1))# final output is 1

model.compile(optimizer="adam", loss="mse")#compiled the model

model.fit(X, y, epochs=200, verbose=1)
new_input = array([130, 160, 180])

new_input = new_input.reshape(1, n_steps, n_features)

new_output = model.predict(new_input, verbose=1)

print (new_output)