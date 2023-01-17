# Temel Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Iris veri seti
dataset = pd.read_csv('../input/iris.csv')

# Aynı sonuçlar için random seed
seed = 7
np.random.seed(seed)
# veri setine bakış
dataset.head()
X = dataset.iloc[:,0:4].values

# Hedef niteliğimiz 4. indekste idi. O sebeple : ile tüm satırları "," den sonra 4 ile hedef niteliğin indeksini seçiyoruz.
y = dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Kategorik olan hedef değişkeni nümerik yap
encoder = LabelEncoder()
y= encoder.fit_transform(y)

# Nümerik hedef değişkenden gölge değişkenler yarat
y = np_utils.to_categorical(y)
from sklearn.model_selection import train_test_split

#Sıralamaya dikkat. Aynı sonuçlar için random_state değeri aynı olmalıdır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 4, units = 4, activation = 'relu'))
classifier.add(Dense(kernel_initializer = 'uniform', units = 3,  activation = 'softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=5, epochs=100)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred.round(), normalize=True)