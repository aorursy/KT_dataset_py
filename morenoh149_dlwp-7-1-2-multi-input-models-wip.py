import tensorflow as tf
layers = tf.keras.layers
Input = tf.keras.Input
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
text_input = Input(shape=(None,), dtype='int32', name='text')
text_input
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
embedded_text
encoded_text = layers.LSTM(32)(embedded_text)
encoded_text
question_input = Input(shape=(None,), dtype='int32', name='question')
question_input
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question],
                                 axis=-1)
answer = layers.Dense(answer_vocabulary_size,
                      activation='softmax')(concatenated)
model = tf.keras.Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['acc'])
model.summary()
'''
# pydotplus is not available in gpu kernels
from keras.utils import plot_model 
plot_model(model, to_file='../input/multi-input-model.png')
'''
from IPython.display import Image
Image(filename='../input/multi-input-model.png') 
import numpy as np
num_samples = 1000
max_length = 100
text = np.random.randint(1,
                        text_vocabulary_size,
                        size=(num_samples, max_length))
question = np.random.randint(1,
                            question_vocabulary_size,
                            size=(num_samples, max_length))
answers = np.random.randint(0,
                           1,
                           size=(num_samples, answer_vocabulary_size))
# fitting using a list of inputs
# model.fit([text, question], answers, epochs=10, batch_size=128)
# fitting using a dictionary of inputs (only if inputs are named)
model.fit({'text': text, 'question': question},
          answers, epochs=10, batch_size=128)
model
text = "A federal judge in Texas is weighing a request by 20 states to suspend the Affordable Care Act (ACA), a move that could lead to chaos in the health insurance market, some industry experts worry."
question = "Who answered the request?"
predict_input = np.array([1,2,3])
predict_input.shape
model.input_shape
prediction = model.predict([predict_input, predict_input])
prediction
sum(prediction[0])