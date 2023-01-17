import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
previsores = pd.read_csv('../input/entradas-breast.csv')
classe = pd.read_csv('../input/saidas-breast.csv')
prev_train, prev_test, class_train, class_test = train_test_split(previsores, classe, test_size=0.25)
classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dense(units=1, activation='sigmoid'))
otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(prev_train, class_train, batch_size=10, epochs=100)
previsoes = classificador.predict(prev_test)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(class_test, previsoes)
matrix = confusion_matrix(class_test, previsoes)
resultado = classificador.evaluate(prev_test, class_test)
pesos0 = classificador.layers[0].get_weights()
pesos00 = pesos0[0]
bias00 = pesos0[1]
pesos1 = classificador.layers[1].get_weights()
pesos01 = pesos1[0]
bias01 = pesos1[1]
pesos2 = classificador.layers[2].get_weights()
pesos02 = pesos2[0]
bias02 = pesos2[1]
