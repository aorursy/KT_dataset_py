import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Train.csv',sep=",")

data = df.loc[:,["Open"]].values



#Del dataset, si considera solo il valore della feature Open. Le restanti feature sono scartate

#Train su tutti gli esempi - gli ultimi 50

train = data[:len(data)-50]

#Test sugli ultimi 50 esempi del dataset

test = data[len(train):]



# reshape

train=train.reshape(train.shape[0],1)



df.head()
print(f'Intervallo feature Open su train: {train.min()} - {train.max()}')

print(f'Intervallo feature Open su test: {test.min()} - {test.max()}')



from sklearn.preprocessing import MinMaxScaler

#Scalatura dell'intervallo di definizione di Open in [0,1]

scaler = MinMaxScaler(feature_range= (0,1))

train_scaled = scaler.fit_transform(train)





plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

plt.title("DataSet prima di MinMaxScaler()")

plt.plot(train)



plt.subplot(1, 2, 2)

plt.title("DataSet dopo MinMaxScaler()")

plt.plot(train_scaled)

plt.show()
# Definizione della finestra di predizione.

X_train = []

y_train = []

window_size = 50



for i in range(window_size, train_scaled.shape[0]):

    X_train.append(train_scaled[i-window_size:i,0])

    y_train.append(train_scaled[i,0])



#Conversione lista -> array

X_train, y_train = np.array(X_train), np.array(y_train)



# Aggiunta della terza dimensione mediante Reshape()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (1158, 50, 1)
from keras.models import Sequential  

from keras.layers import Dense 

from keras.layers import SimpleRNN

from keras.layers import Dropout



# Definizione del modello di RNN

model = Sequential()



#Definizione del primo layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))

model.add(Dropout(0.2)) #Spegne il 20% dei neuroni



#Definizione del secondo layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))

model.add(Dropout(0.2))



#Definizione del terzo layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))

model.add(Dropout(0.2))



#Definizione del quarto layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 50))

model.add(Dropout(0.2))



# Definizione Output Layer

model.add(Dense(units = 1))



# Compilazione della RNN

model.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fit di RNN

#batch_size: numero di esempi di training da prendere in considerazione

model.fit(X_train, y_train, epochs = 100, batch_size = 32)
inputs = data[len(data) - len(test) - window_size:]

#Utilizzo di Min Max Scaler per scalare i dati presenti in inputs

inputs = scaler.transform(inputs)
X_test = []

for i in range(window_size, inputs.shape[0]):

    #Si assumono 50 esempi da 0 a 50, da 1 a 51.

    X_test.append(inputs[i-window_size:i, 0]) 

X_test = np.array(X_test)

#Trasformazione in dimensioni compatibili con il tensore

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
predicted_data = model.predict(X_test)

#Inverse_Transoform: Annulla lo scaling di predicted_data in base al range di definizione della feature

predicted_data = scaler.inverse_transform(predicted_data)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="orange",label="Real value")

plt.plot(predicted_data,color="c",label="RNN predicted result")

plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()
# Import delle librerie

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
model = Sequential()

model.add(LSTM(10, input_shape=(None,1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=50, batch_size=1)
predicted_data2=model.predict(X_test)

predicted_data2=scaler.inverse_transform(predicted_data2)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="LimeGreen",label="Real values")

plt.plot(predicted_data2,color="Gold",label="Predicted LSTM result")

plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="green", linestyle='dashed',label="Valori reali")

plt.plot(predicted_data2,color="blue", label="LSTM predicted result")

plt.plot(predicted_data,color="red",label="RNN predicted result")

plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()
#Import delle librerie

from keras.models import Sequential  

from keras.layers import Dense 

from keras.layers import SimpleRNN

from keras.layers import Dropout 



# Definizione del modello di RNN

model = Sequential()



#Definizione del primo layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 100,activation='relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))

model.add(Dropout(0.2))



#Definizione del secondo layer con l'aggiunta della regolarizzazione mediante Dropout

model.add(SimpleRNN(units = 50))

model.add(Dropout(0.2))





# Definizione Output Layer

model.add(Dense(units = 1)) 



# Compilazione della RNN

model.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fit di RNN

#batch_size: numero di esempi di training da prendere in considerazione

model.fit(X_train, y_train, epochs = 500, batch_size = 16)
predicted_data_modified = model.predict(X_test)

predicted_data_modified = scaler.inverse_transform(predicted_data_modified)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="gray",label="Real values")

plt.plot(predicted_data,color="cyan",label="RNN result v1")

plt.plot(predicted_data_modified,color="blue",label="RNN result v2")



plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()
#Import delle librerie

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler 





model = Sequential()

model.add(LSTM(10, input_shape=(None,1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=200, batch_size=4)
predicted_data2_modified=model.predict(X_test)

predicted_data2_modified=scaler.inverse_transform(predicted_data2_modified)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="DimGray",label="Real values", linestyle="dashed")

plt.plot(predicted_data2,color="Magenta",label="LSTM predicted")

plt.plot(predicted_data2_modified,color="c", label="LSTM v2 predicted")

plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()
plt.figure(figsize=(16,8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(test,color="DimGray",label="Real value", linestyle="dashed")

plt.plot(predicted_data2,color="blue",label="LSTM predicted")

plt.plot(predicted_data2_modified,color="red", linestyle="dashed", label="LSTM Modified predicted")

plt.plot(predicted_data,color="c",label="RNN predicted")

plt.plot(predicted_data_modified,color="green", linestyle="dashed", label="RNN modified predicted")

plt.legend()

plt.xlabel("Giorni")

plt.ylabel("Valore di apertura")

plt.grid(True)

plt.show()