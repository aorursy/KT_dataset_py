import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar100

import keras
from keras.models import Sequential
from keras.layers import *

import utilities
np.set_printoptions(suppress=True)
(X_train, y_train), (X_valid, y_valid) = cifar100.load_data()

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_valid.shape:', X_valid.shape)
print('y_valid.shape:', y_valid.shape)
y_train = y_train.reshape(-1,)
y_valid = y_valid.reshape(-1,)
labels = np.array([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
    'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
    'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard',
    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
    'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 
    'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
])

groups = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}    

sel = np.random.choice(range(50000), 18, replace=False)

plt.figure(figsize=(12,6))

for i in range(18):
  plt.subplot(3, 6, i+1)
  plt.imshow(X_train[sel,:,:,:][i])
  plt.axis('off')
  plt.text(0,-1.2, labels[y_train[sel][i]])

plt.show()
X_train_sc = X_train / 255
X_valid_sc = X_valid / 255
cnn = Sequential()

cnn.add(Conv2D(128, (3,3), input_shape=(32,32,3), activation='relu', padding='same'))
cnn.add(Conv2D(128, (3,3), input_shape=(32,32,3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.25))
cnn.add(BatchNormalization())

cnn.add(Conv2D(256, (3,3), activation='relu', padding='same'))
cnn.add(Conv2D(256, (3,3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.5))
cnn.add(BatchNormalization())

cnn.add(Conv2D(512, (3,3), activation='relu', padding='same'))
cnn.add(Conv2D(512, (3,3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.75))
cnn.add(BatchNormalization())

cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.75))
cnn.add(BatchNormalization())

cnn.add(Dense(100, activation='softmax'))

cnn.summary()
%%time

opt = keras.optimizers.Adam(0.01)

cnn.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

h1 = cnn.fit(X_train_sc, y_train, batch_size=2048, epochs=40, verbose=2, validation_data=(X_valid_sc, y_valid))
utilities.vis_training([h1])
h2 = cnn.fit(X_train_sc, y_train, batch_size=2048, epochs=40, verbose=1, validation_data=(X_valid_sc, y_valid))
utilities.vis_training([h1, h2], start=20)
cnn.save('v03_mod01.h5')
%%time

keras.backend.set_value(cnn.optimizer.lr, 0.001)
h3 = cnn.fit(X_train_sc, y_train, batch_size=2048, epochs=40, verbose=1, validation_data=(X_valid_sc, y_valid))
utilities.vis_training([h1, h2, h3], start=20)
cnn.save('v03_mod02.h5')
valid_prob = cnn.predict_proba(X_valid_sc)
print(valid_prob.shape)
print(np.round(valid_prob[:20,:10], 4))
ranked_classes = np.argsort(-valid_prob, axis=1)
print(ranked_classes[:20, :10])
for k in range(1, 21):
  top_k_acc = np.sum(y_valid.reshape(-1,1) == ranked_classes[:,:k]) / len(y_valid)
  print('Top', k, 'Accuracy:', round(top_k_acc, 3))
n = np.random.choice(range(10000))
img = X_valid_sc[[n],:,:,:]
label = labels[y_valid[n]]

prob = cnn.predict_proba(img)

top_5_classes = np.argsort(prob)[0, -5:]
top_5_probs = np.sort(prob)[0, -5:]

plt.figure(figsize=(7,3))
plt.tight_layout()
plt.subplot(1,2,1)
plt.imshow(img[0, :, :, :])
plt.text(0, -5, label)
plt.axis('Off')

plt.subplot(1,2,2)
plt.barh(labels[top_5_classes], top_5_probs)
plt.tight_layout()
plt.show()



