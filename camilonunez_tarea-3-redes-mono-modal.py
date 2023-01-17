from keras.datasets import cifar10
import numpy as np
from sklearn import preprocessing

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

enc = preprocessing.OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()

y_test = enc.transform(y_test).toarray()

x_train = (x_train.astype('float32')) / 255.0
x_test = (x_test.astype('float32')) / 255.0

print(x_train.shape)
print(x_test.shape)
from keras.layers import Input, Dense, Conv2D, Reshape, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam

### FeatureExtractor Model

input_layer = Input(shape=(32, 32, 3, ))
        
a_one = Conv2D(64, (3,3), activation='relu', padding='same') (input_layer)
a_three = Conv2D(64, (3,3), activation='relu', padding='same') (a_one)
a_five = MaxPooling2D() (a_three)
block_one = Dropout(0.25) (a_five)

b_one = Conv2D(128, (3,3), activation='relu', padding='same') (block_one)
b_three = Conv2D(128, (3,3), activation='relu', padding='same') (b_one)
b_five = MaxPooling2D() (b_three)
block_two = Dropout(0.25) (b_five)

c_one = Conv2D(256, (3,3), activation='relu', padding='same') (block_two)
c_three = Conv2D(256, (3,3), activation='relu', padding='same') (c_one)
c_five = MaxPooling2D() (c_three)
block_three = Dropout(0.5) (c_five)

d_one = Conv2D(512, (3,3), activation='relu', padding='same') (block_three)
d_three = Conv2D(512, (1,1), activation='relu', padding='same') (d_one)
d_five = MaxPooling2D() (d_three)
block_four = Dropout(0.2) (d_five)

flat = Flatten() (block_four)
fc_one = Dense(4096, activation='relu') (flat)
fc_two = Dense(4096, activation='relu') (fc_one)

final = Dense(10, activation='softmax') (fc_two)

FeatureExtractor = Model(input_layer, final)
feature_extractor = Model(input_layer, flat)
#FeatureExtractor.summary()
FeatureExtractor.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

### Train FeatureExtractor Model
FeatureExtractor.fit(x_train, y_train,epochs=100, batch_size=128, shuffle=True,  validation_data=(x_test, y_test))
from keras.utils.vis_utils import plot_model
plot_model(FeatureExtractor, show_shapes=False, show_layer_names=True)
features = feature_extractor.predict(x_train)
features_test = feature_extractor.predict(x_test)

print(features.shape)
print(features_test.shape)
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.callbacks import TensorBoard

input_dim = 2048
#input_dim = 10
encoding_dim = 128

# DeepAutoencoder Models
input_layer = Input(shape=(input_dim,))

hidden_one = Dense(encoding_dim*4, activation='relu') (input_layer)
hidden_two = Dense(encoding_dim*2, activation='relu') (hidden_one)
encoder_output = Dense(encoding_dim, activation='relu') (hidden_two)
encoder_model = Model(input_layer, encoder_output)

hidden_three = Dense(encoding_dim*2, activation='relu') (encoder_output)
hidden_four = Dense(encoding_dim*4, activation='relu') (hidden_three)
decoder_output = Dense(input_dim, activation='sigmoid') (hidden_four)
autoencoder = Model(input_layer, decoder_output)

autoencoder.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')

### Train DeepAutoencoder Model
autoencoder.fit(features, features,epochs=50, batch_size=256, shuffle=True, validation_data=(features_test, features_test))
plot_model(autoencoder, show_shapes=True, show_layer_names=True)
features = feature_extractor.predict(x_train)
print(features.shape)
vectors = encoder_model.predict(features)
print(vectors.shape)
import numpy as np
from annoy import AnnoyIndex

index = AnnoyIndex(vectors.shape[1],metric='angular')
for i in range(vectors.shape[0]):
    index.add_item(i, vectors[i,:].tolist())
    
index.build(20)
index.save("index.ann")
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
lookup = np.argmax(y_train[:, :], axis=1)
import random
choice_idx = random.randrange(20000)
results = index.get_nns_by_item(choice_idx, 10)

import matplotlib.pyplot as plt
plt.figure(figsize=(1,4))
plt.imshow(x_train[choice_idx])
plt.axis('off')
plt.title('query: ' + str(labels[lookup[choice_idx]]))

plt.figure(figsize=(10,4))
for i in range(10):
    ax = plt.subplot(1, 10, i+1)
    ax.imshow(x_train[results[i]])
    ax.axis('off')
    ax.set_title(str(labels[lookup[results[i]]]))

plt.show()