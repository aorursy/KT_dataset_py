import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout, Flatten
import matplotlib.pyplot as plt

# To generate the text
def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

data = open('../input/tarantino_scripts.txt', 'r').read()
chars = sorted(list(set(data)))
VOCAB_SIZE = len(chars)
SEQ_LENGTH = 50

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

X = np.zeros((int(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((int(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, int(len(data) / SEQ_LENGTH)):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence
    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

print(X.shape, y.shape)

model = Sequential()
model.add(LSTM(1024, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(LSTM(1024, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

losses = []

for nbepoch in range(1, 151):
    print('Epoch ', nbepoch)
    history = model.fit(X, y, batch_size=64, verbose=1, epochs=1)
    if nbepoch % 10 == 0:
        model.model.save('checkpoint_{}_epoch_{}.h5'.format(512, nbepoch))
    generate_text(model, 50)
    print("\nLoss is ", history.history['loss'])
    losses.append(history.history['loss'][-1])
    print('\n\n\n')

model.save('final_model.h5')

with open("losses.txt", "w") as text_file:
    s = ""
    index = 1
    for loss in losses:
        s = s + "Epoch " + str(index) + "- Loss is " + str(loss) + "\n"
        index = index + 1
    text_file.write(s)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_graph.png')
plt.show()
