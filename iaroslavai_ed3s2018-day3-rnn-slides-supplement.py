import numpy as np

# support function
randm = lambda n, m: np.random.randn(n, m)
randi = lambda : np.random.randint(3, 8)

N = 1000

# set of inputs: sequences, or 2d matricies
X = [randm(randi(), 2) for _ in range(N)]
y = [1*(x[-1, -1] > 0) for x in X]

print('One example of input and output:')
display(X[0])
display(y[0])
from sklearn.ensemble import GradientBoostingClassifier

# make a data preprocessing function
def pad_seq(x):
    y = np.zeros((8, 2))
    y[:len(x)] = x
    return y

def flatten(x):
    return x.ravel()

print('Original:')
display(X[0])
print('Preprocessed:')
display(flatten(pad_seq(X[0])))
Xp = [flatten(pad_seq(x)) for x in X]

model = GradientBoostingClassifier()
model.fit(Xp, y)
model.score(Xp, y)
from keras import Sequential
from keras.layers import InputLayer, Flatten, Dense, LeakyReLU, Activation
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

# no flattening
Xp = [pad_seq(x) for x in X]

# make proper formats
Xp = np.array(Xp)
y = np.array(y)

# Feed forward model with a special "Flatten" layer
model = Sequential(layers=[
    InputLayer(input_shape=(8, 2)),
    Flatten(),
    Dense(64),
    LeakyReLU(),
    Dense(2),
    Activation('softmax')
])

# compile the training and evaluation routines of the model
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(lr=0.1),
    metrics=['acc']
)

# fit the model
model.fit(Xp, y, epochs=300, verbose=0, batch_size=128)

# evaluate the model, returns the tuple (loss, accuracy)
model.evaluate(Xp, y)