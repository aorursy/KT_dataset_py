import tensorflow_datasets as tfds

import tensorflow as tf



tfds.disable_progress_bar()
dataset, info = tfds.load('imdb_reviews/subwords8k', 

                          with_info=True,

                          as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
encoder = info.features['text'].encoder

print('Vocabulary Size: ', encoder.vocab_size)
sample_string = 'Hello Dear Kegglers :) '



encoded_string = encoder.encode(sample_string)

print('Encoded string is {}'.format(encoded_string))



original_string = encoder.decode(encoded_string)

print('The original string: "{}"'.format(original_string))
assert original_string == sample_string

for index in encoded_string:

  print('{} ----> {}'.format(index, encoder.decode([index])))
BUFFER_SIZE = 10000

BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)



test_dataset = test_dataset.padded_batch(BATCH_SIZE)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(encoder.vocab_size, 64),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1)

])



model.summary()
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

def build_lrfn(lr_start=5e-5, lr_max=1e-4, 

               lr_min=0, lr_rampup_epochs=12, 

               lr_sustain_epochs=2, lr_exp_decay=.8):



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    return lrfn



lrfn = build_lrfn()

lr_schedule = LearningRateScheduler(lrfn, verbose=True)



checkpoint1 = ModelCheckpoint(

    filepath='best_weights_single_LSTM.hdf5',

    save_weights_only=True,

    monitor='val_accuracy',

    mode='max',

    save_best_only=True)



checkpoint2 = ModelCheckpoint(

    filepath='best_weights_double_LSTM.hdf5',

    save_weights_only=True,

    monitor='val_accuracy',

    mode='max',

    save_best_only=True)



callbacks1 = [lr_schedule, checkpoint1]

callbacks2 = [lr_schedule, checkpoint2]
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(),

              metrics=['accuracy'])
train_history1 = model.fit(train_dataset, epochs=20,

                    validation_data=test_dataset,

                    callbacks = callbacks1,

                    validation_steps=100)
def visualize_training(history, lw = 3):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))

    plt.subplot(2,1,1)

    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)

    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)

    plt.title('Accuracy Comparison')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.grid(True)

    plt.legend(fontsize = 'x-large')

    



    plt.subplot(2,1,2)

    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)

    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)

    plt.title('Loss Comparison')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend(fontsize = 'x-large')

    plt.grid(True)

    plt.show()



    plt.figure(figsize=(10,5))

    plt.plot(history.history['lr'], label = 'lr', marker = '*',linewidth = lw)

    plt.title('Learning Rate')

    plt.xlabel('Epochs')

    plt.ylabel('Learning Rate')

    plt.grid(True)

    plt.show()
visualize_training(train_history1) 
test_loss, test_acc = model.evaluate(test_dataset)



print('Test Loss: {}'.format(test_loss))

print('Test Accuracy: {}'.format(test_acc))
def pad_to_size(vec, size):

    zeros = [0] * (size - len(vec))

    vec.extend(zeros)

    return vec

# conditional encoder. It encodes with padding 

def sample_predict(sample_pred_text, pad):

    encoded_sample_pred_text = encoder.encode(sample_pred_text)



    if pad:

        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)

    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)

    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))



    return predictions
# predict on a sample text without padding.



sample_pred_text = ('The movie was good. overall it was a hit movie '

                    'I would recomment watching this movie. ')



print('\n Before adding the complex phrase: ')

print(sample_pred_text)

predictions = sample_predict(sample_pred_text, pad=False)

print('Prediction without padding: ', predictions)



predictions = sample_predict(sample_pred_text, pad=True)

print('Prediction with padding: ', predictions)





# After we add the complex phrase

print('\n\nAfter adding the complex phrase: ')

sample_pred_text = ('The movie was good.  Though the movie lacks in minor details, '

                    'overall it was a hit movie '

                    'I would recomment watching this movie. ')

print(sample_pred_text)



predictions = sample_predict(sample_pred_text, pad=False)

print('Prediction without padding: ', predictions)



predictions = sample_predict(sample_pred_text, pad=True)

print('Prediction with padding: ', predictions)
westworld_review = ('What a miserable letdown. A rambling piece of garbage. '

                    'Westworld started out great with the actual reason the book was written for. '

                    'Fantasmical worlds for the rich. It ended up being car chases in LA somewhere.')



predictions = sample_predict(westworld_review, pad=False)

print('Prediction without padding: ', predictions)



predictions = sample_predict(westworld_review, pad=True)

print('Prediction with padding: ', predictions)
westworld_review2 = ('I really enjoyed the 1st season of Westworld and I think '

                     'it has the best special effects on television')



predictions = sample_predict(westworld_review2, pad=False)

print('Prediction without padding: ', predictions)



predictions = sample_predict(westworld_review2, pad=True)

print('Prediction with padding: ', predictions)
westworld_review3 = ('started to get lost in season #2 and now I am totally lost on season #3.' 

                     ' I would have preferred they spend the money on another season of game of thrones. ' 

                     'I really enjoyed the 1st season of Westworld and I think it has the best special effects on television but')



predictions = sample_predict(westworld_review3, pad=False)

print('Prediction without padding: ', predictions)



predictions = sample_predict(westworld_review3, pad=True)

print('Prediction with padding: ', predictions)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(encoder.vocab_size, 64),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)

])

model.summary()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              optimizer=tf.keras.optimizers.Adam(),

              metrics=['accuracy'])
train_history2 = model.fit(train_dataset, epochs=20,

                    validation_data=test_dataset,

                    callbacks = callbacks2,

                    validation_steps=30)

visualize_training(train_history2)