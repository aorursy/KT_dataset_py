import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense 

from keras.optimizers import Adam, SGD



%matplotlib inline

import matplotlib.pyplot as plt
path = "../input/beer-consumption-sao-paulo/Consumo_cerveja.csv"

dataset = pd.read_csv(path)
dataset.head()
dataset.columns=["fecha", "media", "min", "max", "precipitacao", "finde", "cerveza"]
dataset = dataset.dropna()

dataset.head()
media= np.array(dataset.media)
print(type(media[1]))
media = media.tolist()

print(type(media))
media_2 = []

for i in media:

    media_2.append(float(str(i).replace(",", ".")))
type(media_2)
X=np.array(media_2)

y_true=dataset[['cerveza']].values
model = Sequential()

model.add(Dense(1, input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X,y_true, epochs=35, batch_size=110)
y_pred= model.predict(X)
plt.scatter(X,y_true)

plt.plot(X, y_pred, color='red', linewidth=3)
w,b=model.get_weights()
print("Valor de w:",w)

print("Valor de b:",b)
Xnew = np.array([[15.]])

# make a prediction

ynew = model.predict(Xnew)

# show the inputs and predicted outputs

print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
X
y_pred

#dataset = pd.DataFrame({'Temperatura Media (C)':X, 'Consumo de cerveja (litros)':y_pred})
X.shape
y_pred = np.squeeze(y_pred)
y_pred.shape
dataset = pd.DataFrame({'Temperatura Media (C)':X, 'Consumo de cerveja (litros)':y_pred})
dataset.head()
dataset.to_csv('predictions.csv' , index=False)
ls