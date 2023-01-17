from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
dataset = pd.read_csv("../input/noticias-falsas-en-espaol/fakes1000.csv")
print(dataset.head(10))
print(dataset.columns.values)
print(dataset.info())
print(dataset.describe())
texts = []
classes = []
for i, label in enumerate(dataset['class']):
    texts.append(dataset['Text'][i])
    if label == 'TRUE':
        classes.append(0)
    else:
        classes.append(1)
        
texts = np.asarray(texts)
classes = np.asarray(classes)
maxFeatures = 14000
maxLen = 1000

trainingData = int(len(texts) * .8)
validationData = int(len(texts) - trainingData)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Se encontaron {0} palabras únicas: ".format(len(word_index)))
data = pad_sequences(sequences, maxlen=maxLen)
print("Forma de los datos: ", data.shape)

np.random.seed(42)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = classes[indices]

X_train = data[:trainingData]
y_train = labels[:trainingData]
X_test = data[trainingData:]
y_test = labels[trainingData:]
import numpy as np                                       
from keras.preprocessing import sequence                 
from keras.models import Sequential                      
from keras.layers import Dense, Dropout, Activation
network = Sequential()
network.add(Embedding(maxFeatures, 64))
network.add(LSTM(64))
network.add(Dense(1, activation='sigmoid'))

network.add(Dropout(0.2))
network.add(Activation('relu'))
network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = network.fit(X_train, y_train, epochs=10, batch_size=1000, validation_split=0.2)
print("Accuracy (entrenamiento):",history.history['acc'])
pred = network.predict_classes(X_test)
acc = network.evaluate(X_test, y_test)
proba_rnn = network.predict_proba(X_test)
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])

plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss count')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()