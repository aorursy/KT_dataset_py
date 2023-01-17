# Load packages

import numpy as np



from tensorflow.keras.layers import Embedding, Flatten, Input

from tensorflow.keras.models import Model, Sequential
# Define an embedding matrix

EMBEDDING_SIZE = 4

VOCAB_SIZE = 10



embedding_matrix = np.arange(EMBEDDING_SIZE * VOCAB_SIZE, dtype='float32')

embedding_matrix = embedding_matrix.reshape(VOCAB_SIZE, EMBEDDING_SIZE)

print(embedding_matrix)
idx = 3

print(embedding_matrix[idx])
def onehot_encode(dim, label):

    return np.eye(dim)[label]



onehot_idx = onehot_encode(VOCAB_SIZE, idx)

print(onehot_idx)
print(np.dot(onehot_idx, embedding_matrix))
# Define an embedding layer

embedding_layer = Embedding(output_dim=EMBEDDING_SIZE, input_dim=VOCAB_SIZE, weights=[embedding_matrix], input_length=1, name='My_embedding')
# Define a Keras model using this embedding

x = Input(shape=[1], name='Input')

embedding = embedding_layer(x)

model = Model(inputs=x, outputs=embedding)
model.output_shape
model.get_weights()
model.summary()
labels_to_encode = np.array([[3]])

model.predict(labels_to_encode)
labels_to_encode = np.array([[3], [3], [0], [9]])

model.predict(labels_to_encode)
x = Input(shape=[1], name='Input')

y = Flatten()(embedding_layer(x))

model2 = Model(inputs=x, outputs=y)
model2.output_shape
model2.predict(np.array([3]))
model2.summary()
model2.set_weights([np.ones(shape=(VOCAB_SIZE, EMBEDDING_SIZE))])
labels_to_encode = np.array([[3]])

model2.predict(labels_to_encode)
model.predict(labels_to_encode)
model3 = Sequential([embedding_layer, Flatten()])
model3.predict(labels_to_encode)