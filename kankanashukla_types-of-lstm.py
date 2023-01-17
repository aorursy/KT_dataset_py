# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a univariate sequence into samples

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



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=200, verbose=0)

# demonstrate prediction

x_input = array([70, 80, 90])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# univariate stacked lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a univariate sequence

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



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=200, verbose=0)

# demonstrate prediction

x_input = array([70, 80, 90])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# univariate bidirectional lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

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



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=200, verbose=0)

# demonstrate prediction

x_input = array([70, 80, 90])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# univariate cnn lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D



# split a univariate sequence into samples

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



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps = 4

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

n_features = 1

n_seq = 2

n_steps = 2

X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

# define model

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))

model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=500, verbose=0)

# demonstrate prediction

x_input = array([60, 70, 80, 90])

x_input = x_input.reshape((1, n_seq, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# multivariate data preparation

from numpy import array

from numpy import hstack



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps

                # check if we are beyond the dataset

                if end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)

# choose a number of time steps

n_steps = 3

# convert into input/output

X, y = split_sequences(dataset, n_steps)

print(X.shape, y.shape)

# summarize the data

for i in range(len(X)):

        print(X[i], y[i])
# multivariate lstm example

from numpy import array

from numpy import hstack

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps

                # check if we are beyond the dataset

                if end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)

# choose a number of time steps

n_steps = 3

# convert into input/output

X, y = split_sequences(dataset, n_steps)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=200, verbose=0)

# demonstrate prediction

x_input = array([[80, 85], [90, 95], [100, 105]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# multivariate output data prep

from numpy import array

from numpy import hstack



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps

                # check if we are beyond the dataset

                if end_ix > len(sequences)-1:

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)

# choose a number of time steps

n_steps = 3

# convert into input/output

X, y = split_sequences(dataset, n_steps)

print(X.shape, y.shape)

# summarize the data

for i in range(len(X)):

        print(X[i], y[i])
# multivariate output stacked lstm example

from numpy import array

from numpy import hstack

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps

                # check if we are beyond the dataset

                if end_ix > len(sequences)-1:

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

print(dataset)

# choose a number of time steps

n_steps = 3

# convert into input/output

X, y = split_sequences(dataset, n_steps)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]

# define model

model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=400, verbose=0)

# demonstrate prediction

x_input = array([[70,75,145], [80,85,165], [90,95,185]])

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# multi-step data preparation

from numpy import array



# split a univariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequence)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out

                # check if we are beyond the sequence

                if out_end_ix > len(sequence):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# split into samples

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# summarize the data

for i in range(len(X)):

        print(X[i], y[i])
# univariate multi-step vector-output stacked lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a univariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequence)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out

                # check if we are beyond the sequence

                if out_end_ix > len(sequence):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# split into samples

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

print(X)

# define model

model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=50, verbose=0)

# demonstrate prediction

x_input = array([70, 80, 90])

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# univariate multi-step encoder-decoder lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import RepeatVector

from keras.layers import TimeDistributed



# split a univariate sequence into samples

def split_sequence(sequence, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequence)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out

                # check if we are beyond the sequence

                if out_end_ix > len(sequence):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# split into samples

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

print(X)

y = y.reshape((y.shape[0], y.shape[1], n_features))

print(y)

# define model

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))

model.add(RepeatVector(n_steps_out))

model.add(LSTM(100, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=100, verbose=0)

# demonstrate prediction

x_input = array([70, 80, 90])

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# multivariate multi-step data preparation

from numpy import array

from numpy import hstack



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out-1

                # check if we are beyond the dataset

                if out_end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# covert into input/output

X, y = split_sequences(dataset, n_steps_in, n_steps_out)

print(X.shape, y.shape)

# summarize the data

for i in range(len(X)):

        print(X[i], y[i])
# multivariate multi-step stacked lstm example

from numpy import array

from numpy import hstack

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out-1

                # check if we are beyond the dataset

                if out_end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# covert into input/output

X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]

# define model

model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=200, verbose=0)

# demonstrate prediction

x_input = array([[70, 75], [80, 85], [90, 95]])

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# multivariate multi-step data preparation

from numpy import array

from numpy import hstack

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import RepeatVector

from keras.layers import TimeDistributed



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out

                # check if we are beyond the dataset

                if out_end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# covert into input/output

X, y = split_sequences(dataset, n_steps_in, n_steps_out)

print(X.shape, y.shape)

# summarize the data

for i in range(len(X)):

        print(X[i], y[i])
# multivariate multi-step encoder-decoder lstm example

from numpy import array

from numpy import hstack

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import RepeatVector

from keras.layers import TimeDistributed



# split a multivariate sequence into samples

def split_sequences(sequences, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequences)):

                # find the end of this pattern

                end_ix = i + n_steps_in

                out_end_ix = end_ix + n_steps_out

                # check if we are beyond the dataset

                if out_end_ix > len(sequences):

                        break

                # gather input and output parts of the pattern

                seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

                X.append(seq_x)

                y.append(seq_y)

        return array(X), array(y)



# define input sequence

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])

in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])

out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure

in_seq1 = in_seq1.reshape((len(in_seq1), 1))

in_seq2 = in_seq2.reshape((len(in_seq2), 1))

out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns

dataset = hstack((in_seq1, in_seq2, out_seq))

# choose a number of time steps

n_steps_in, n_steps_out = 3, 2

# covert into input/output

X, y = split_sequences(dataset, n_steps_in, n_steps_out)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]

# define model

model = Sequential()

model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))

model.add(RepeatVector(n_steps_out))

model.add(LSTM(200, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(n_features)))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=300, verbose=0)

# demonstrate prediction

x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)