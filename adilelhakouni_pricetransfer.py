# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

! ls /kaggle/input
import pandas as pd
import numpy as np

cars_data = pd.read_csv('/kaggle/input/carsnyd/cars_data_final.csv')

cars_data['Kilometers_Driven'] = cars_data['Kilometers_Driven'].astype('category')
cars_data.head()
cars_data= cars_data.drop(['Unnamed: 0'], axis=1)
cars_data.info()
Y = pd.DataFrame(cars_data['Price'])
X = cars_data.drop(['Price'], axis=1)
print(X.head())
# import sklearn.tree
# import sklearn.neighbors
# from sklearn.svm import SVC # "Support Vector Classifier"
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import neural_network

# # On va définir plusieurs modèles pour voir lequel est le meilleur
# # logistic_m = LogisticRegression() #régression logistique
# # tree_m = sklearn.tree.DecisionTreeClassifier(max_depth=3) #arbre de décision
# # gradient_descent_m=SGDClassifier()# stochastic gradient descent
# # gradient_boosting_m=GradientBoostingClassifier()# gradient boosting
# # knn_m = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3) #k nearest neighbors
# # svm_m = SVC(kernel='linear') # Support Vector Classifier (inspiré du SVM)
# neural_net_MM = sklearn.neural_network.MLPRegressor(
#     hidden_layer_sizes=(12,6,),  activation='relu', solver='adam', alpha=0.002, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000, shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08) #Réseau de neurones, on a ici spécifié tous les paramètres du réseau (possible avec n'importe quel modèle)


# #on entraine nos modèles avec la méthode .fit()
# # model1 = logistic_m.fit(X_train, y_train)
# # model2 = tree_m.fit(X_train, y_train)
# # model3 = gradient_descent_m.fit(X_train, y_train)
# # model4 = gradient_boosting_m.fit(X_train, y_train)
# # model5 = knn_m.fit(X_train, y_train)
# # model6 = svm_m.fit(X_train, y_train)
# model7 = neural_net_MM.fit(X_train, y_train)
### The NYD data
nyd_data = pd.read_csv('/kaggle/input/adilnyd/Adil-data.csv',sep=';')
nyd_data["coef d'état"] = nyd_data["coef d'état"].replace(',','.')
nyd_data["coeff dgamme"] = nyd_data["coeff dgamme"].str.replace(',','.')
nyd_data["PRIX"] = nyd_data["PRIX"].str.replace(',','.')

cols = nyd_data.select_dtypes(exclude=['float','int']).columns

nyd_data[cols] = nyd_data[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
nyd_data["COTE"] = nyd_data["COTE"].astype('float32')
nyd_data["coef d'état"] = nyd_data["coef d'état"].astype('category')

nyd_data=nyd_data[["ANNEE","coef d'état","PRIX","COTE","coeff dgamme"]]
nyd_data.columns=cars_data.columns
nyd_data.info()
Y_train_cars = Y
X_train_cars = X

# NYD
Y_train_nyd = np.array(nyd_data['Price'][:-10]).astype(np.float32)
Y_test_nyd = np.array(nyd_data['Price'][-10:]).astype(np.float32)

X_train_nyd = np.array(nyd_data.drop(["Price"],axis=1).iloc[:-10,:]).astype(np.float32)
X_test_nyd = np.array(nyd_data.drop(["Price"],axis=1).iloc[-10:,:]).astype(np.float32)


Y_train_nyd
X_train_cars=np.array(X_train_cars).astype(np.float32)
Y_train_cars=np.array(Y_train_cars).astype(np.float32)
#mse = tf.keras.losses.MeanSquaredError()
#Deep learning library : Tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.core import Activation, Reshape, Permute

from tensorflow import keras
from tensorflow.keras import layers



def cars_model():
    model= keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(4,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
      ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

   

    return model

model_cars=cars_model()

print('start learning')
EPOCHS = 2000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Fitting the model with the cars data

history = model_cars.fit(X_train_cars, Y_train_cars,
               epochs=EPOCHS, validation_split = 0.2, verbose=0, 
               callbacks=[early_stop])

print(history.history)

fig=plt.figure()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Weights of the cars model 
cars_weights = model_cars.get_weights()
#Create transfer learning model
transf_model =cars_model()
#Set transfer model weights
transf_model.set_weights(cars_weights)
#Set all layers trainable to False (except final conv layer)
for layer in transf_model.layers:
    layer.trainable = False
transf_model.layers[5].trainable = True
print(transf_model.layers[5])
#Compile model
transf_model.compile(loss='mse', optimizer='adam')
#Train model on second part of the data
transf_model.fit(X_train_nyd,Y_train_nyd, verbose=2)
#Store transfer model weights
transf_weights = transf_model.get_weights()

#Check where the weights have changed
# for i in range(len(cars_weights)):
#     update_w = np.sum(cars_weights[i] != transf_weights[i])
#     if update_w != 0:
#         print(str(update_w)+' updated weights for layer '+str(transf_model.layers[i]))


#loss, mse = transf_model.evaluate(X_test_nyd, Y_test_nyd, verbose=2)

#print("Testing set Mean Abs Error: {:5.2f} MPG".format(mse))
transf_model.summary()
predictions7 = transf_model.predict(X_test_nyd)

from sklearn.metrics import mean_squared_error as msr

rmse7 = msr(Y_test_nyd,predictions7)
print(rmse7)
print(predictions7)
cars_weights
import matplotlib.pyplot as plt

ax = pd.DataFrame(predictions7, index=range(0,10), columns=['forecast']).plot(figsize=(20, 10))
ax.set_title('Prédiction des ventes des voitures ')

pd.DataFrame(Y_test_nyd).plot(ax=ax)

plt.show
#plt.savefig('cars_forcast.png')