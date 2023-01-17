import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
embedding_layer = layers.Embedding(1000, 5)
# Here inside the Embedding(), the first number is the input_size and second one is the output_size
result = embedding_layer(tf.constant([1,2,3, 999]))
result.numpy()
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
print(result.shape)
print(result[0])
print(result[1])
# (train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', 
#                                           split = (tfds.Split.TRAIN, tfds.Split.TEST),
#                                           with_info=True, 
#                                           as_supervised=True)

(train_data, test_data), info = tfds.load('imdb_reviews/subwords32k', 
                                          split = (tfds.Split.TRAIN, tfds.Split.TEST),
                                          with_info=True, 
                                          as_supervised=True)
encoder = info.features['text'].encoder
encoder.subwords[50:70]
train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)
train_batch, train_labels = next(iter(train_batches))
print('Training data info')
print('Train data shape: ', train_batch.shape)
print('Train_data: ', train_batch.numpy())


print('\n\nTrain label info')
print('Train label shape: ', train_labels.shape)
print('Train label: ', train_labels.numpy())

from IPython.display import YouTubeVideo
YouTubeVideo('hjx-zwVdfjc', width=800, height=450)

YouTubeVideo('upto_vdrXFI', width=800, height=450)
embedding_dim = 32

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
#   layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_batches,
                    epochs=5,
                    validation_data=test_batches, 
                    validation_steps=250)
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

#     plt.figure(figsize=(10,5))
#     plt.plot(history.history['lr'], label = 'lr', marker = '*',linewidth = lw)
#     plt.title('Learning Rate')
#     plt.xlabel('Epochs')
#     plt.ylabel('Learning Rate')
#     plt.grid(True)
#     plt.show()
visualize_training(history)
# Retrieving the layer
e = model.layers[0]
#Retrieving the weights learned in that layer. 
weights = e.get_weights()[0]
# The weight matrix is basically a list type matrix. Let's convert them into a numpy array for easier visualization.
print(weights.shape)
import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num+1] # skip 0, it's padding.
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
YouTubeVideo('hjx-zwVdfjc', width=800, height=450)

def get_location(word, vocab):
    location = None
    for index, s in enumerate(vocab):
        if(s == word):
            location = index
            break
    return location
vocab = encoder.subwords

man = weights[get_location("man", vocab)]
woman = weights[get_location("woman", vocab)]
boy = weights[get_location("boy", vocab)]
girl = weights[get_location("girl", vocab)]
what = boy - man + woman
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print('Similarity between   man & what  : ', np.squeeze(cosine_similarity([what], [man])))
print('Similarity between   woman & what: ', np.squeeze(cosine_similarity([what], [woman])))
print('Similarity between   boy & what  : ', np.squeeze(cosine_similarity([what], [boy])))
print('Similarity between   girl & what : ', np.squeeze(cosine_similarity([what], [girl])))
