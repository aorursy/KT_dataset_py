import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split # Aufteilung in Training und Test
mnist = fetch_mldata('MNIST original') # images of hand written digits
from sklearn.model_selection import train_test_split # Aufteilung in Training und Test
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance
import matplotlib.pyplot as plt

train_x, test_x, train_y, test_y = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)
scaler = StandardScaler()
scaler.fit(train_x) # Skalierung wird berechnet
train_x2 = scaler.transform(train_x)
test_x2 = scaler.transform(test_x)
from keras.models import Sequential
from keras.layers import Dense,Reshape,Dropout
from keras.optimizers import Adam

model_conf = [32,32, 10]
def make_model(conf, inp_shape):
    model = Sequential()
    inp = inp_shape
    for idx, units in enumerate(conf):
        activation = 'softmax' if idx == len(conf) - 1 else 'relu'
        model.add(Dense(units, input_shape=(inp,), activation=activation))
        inp = units
    opt = Adam(lr=0.0005)
    model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model
    
model = make_model(model_conf, train_x.shape[1])
model.summary()
history = model.fit(train_x2, train_y, epochs=75, batch_size=128, verbose=0)
scores = model.evaluate(test_x2, test_y)
print("Accuracy: %d%%" % (scores[1]*100))
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(n_components=0.95) # Die Varianz der Daten muss groesser 95% sein
pca.fit(train_x2)
train_x2_pca = pca.transform(train_x2)
test_x2_pca = pca.transform(test_x2)
model_pca = make_model(model_conf, train_x2_pca.shape[1])
history_pca = model_pca.fit(train_x2_pca, train_y, epochs=75, batch_size=128, verbose=0)
scores_pca = model_pca.evaluate(test_x2_pca, test_y)
print("Accuracy: %d%%" % (scores_pca[1]*100))
plt.plot(history_pca.history['acc'])
plt.title('Model w/ PCA accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history_pca.history['loss'])
plt.title('Model w/ PCA loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()