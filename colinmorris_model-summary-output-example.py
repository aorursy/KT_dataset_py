from tensorflow import keras

user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')
user_embedding_size = movie_embedding_size = 8
user_embedded = keras.layers.Embedding(1000, user_embedding_size, 
                                       input_length=1, name='user_embedding')(user_id_input)
movie_embedded = keras.layers.Embedding(1000, movie_embedding_size, 
                                        input_length=1, name='movie_embedding')(movie_id_input)
concatenated = keras.layers.Concatenate()([user_embedded, movie_embedded])
out = keras.layers.Flatten()(concatenated)
out = keras.layers.Dense(1)(out)
model = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)
model.summary()
model.summary(line_length=88)