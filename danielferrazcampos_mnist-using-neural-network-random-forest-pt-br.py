import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import time

import random



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from mlxtend.plotting import plot_confusion_matrix

from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials



SEED = 42



random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)
def pred(model, x_test):

    pred_prob = model.predict(x_test)

    pred = np.argmax(pred_prob, axis = 1)

    return pred





def plot_confusion_mtx(model, x_test, y_test, plot_tittle):

    pred_prob = model.predict(x_test)

    pred = np.argmax(pred_prob, axis = 1)



    CM = confusion_matrix(y_test, pred)



    plot_confusion_matrix(conf_mat = CM, figsize = (16, 8))

    plt.title(plot_tittle)

    plt.xticks(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    plt.yticks(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



    plt.show()

    

def plot_confusion_mtx2(model, x_test, y_test, plot_tittle):

    pred= model.predict(x_test)



    CM = confusion_matrix(y_test, pred)



    plot_confusion_matrix(conf_mat = CM, figsize = (16, 8))

    plt.title(plot_tittle)

    plt.xticks(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    plt.yticks(range(10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



    plt.show()
(trainX, trainy), (testX, testy) = mnist.load_data()

reshaped_trainX_unscaled = trainX.reshape([trainX.shape[0], -1]).astype('float32')

reshaped_testX_unscaled = testX.reshape([testX.shape[0], -1]).astype('float32')



scaler = MinMaxScaler()



reshaped_trainX = scaler.fit_transform(reshaped_trainX_unscaled)

reshaped_testX = scaler.fit_transform(reshaped_testX_unscaled)



encoded_trainy = to_categorical(trainy)

encoded_testy = to_categorical(testy)



print('\nFormato train dataset:\t%s\nTrain dataset reshaped:\t%s\nFormato labels dataset:\t%s' % (trainX.shape, reshaped_trainX.shape, trainy.shape))

print('\nFormato test dataset:\t%s\nTest dataset reshaped:\t%s\nFormato labels dataset:\t%s' % (testX.shape, reshaped_testX.shape, testy.shape))
plt.rcParams.update({'font.size': 16})



fig = plt.figure(figsize = (6, 6))

columns = 4

rows = 3



for i in range(1, columns * rows + 1):

    rnd = np.random.randint(0, len(trainX))

    img = trainX[rnd]  

    fig.add_subplot(rows, columns, i)

    plt.title(trainy[rnd])

    plt.axis('off')

    plt.imshow(img, cmap='gray')



plt.show()
NN = Sequential(name = 'Simple_NN')

NN.add(Dense(512, input_dim=784, activation='relu', name='input_layer'))

NN.add(Dense(10, activation='softmax', name='output_layer'))



print("input shape ",NN.input_shape)

print("output shape ",NN.output_shape)
NN.compile(loss=tf.keras.losses.categorical_crossentropy, 

           optimizer=tf.keras.optimizers.Adam(), 

           metrics=['accuracy'])



NN.summary()
start = time.time()

history = NN.fit(reshaped_trainX, encoded_trainy, epochs=50, batch_size=64, verbose=0)



predicted = pred(NN, reshaped_testX)



elapsed_time = time.time() - start



NN_acc = accuracy_score(testy, predicted)

NN_time = elapsed_time



print("O treinamento levou %.0f segundos.\nAcurácia: %.4f" % (elapsed_time, NN_acc))
print ("Reporte de classificação:\n")

print(classification_report(testy, predicted))
plot_confusion_mtx(NN, reshaped_testX, testy, 'Rede Neural Simples')
start = time.time()

history = NN.fit(reshaped_trainX_unscaled, encoded_trainy, epochs=50, batch_size=64, verbose=0)



predicted = pred(NN, reshaped_testX_unscaled)



elapsed_time = time.time() - start



NN_acc_ns = accuracy_score(testy, predicted)

NN_time_ns = elapsed_time



print("O treinamento levou %.0f segundos.\nAcurácia: %.4f" % (elapsed_time, NN_acc_ns))
print ("Reporte de classificação:\n")

print(classification_report(testy, predicted))
plot_confusion_mtx(NN, reshaped_testX, testy, 'Rede Neural Simples (Dados Não-Normalizados)')
DNN = Sequential(name = 'Deep_NN')

DNN.add(Dense(512, input_dim=784, activation='relu', name='input_layer'))

DNN.add(Dense(256, activation='relu', name='hidden_layer1'))

DNN.add(Dense(128, activation='relu', name='hidden_layer2'))

DNN.add(Dense(64, activation='relu', name='hidden_layer3'))

DNN.add(Dense(32, activation='relu', name='hidden_layer4'))

DNN.add(Dense(10, activation='softmax', name='output_layer'))



print("input shape ",DNN.input_shape)

print("output shape ",DNN.output_shape)
DNN.compile(loss=tf.keras.losses.categorical_crossentropy, 

            optimizer=tf.keras.optimizers.Adam(), 

            metrics=['accuracy'])



DNN.summary()
start = time.time()

history = DNN.fit(reshaped_trainX, encoded_trainy, epochs=50, batch_size=64, verbose=0)



predicted = pred(DNN, reshaped_testX)



elapsed_time = time.time() - start



DNN_acc = accuracy_score(testy, predicted)

DNN_time = elapsed_time



print("O treinamento levou %.0f segundos.\nAcurácia: %.4f" % (elapsed_time, DNN_acc))
print ("Reporte de classificação:\n")

print(classification_report(testy, predicted))
plot_confusion_mtx(DNN, reshaped_testX, testy, 'Rede Neural Profunda')
rf = RandomForestClassifier(max_depth = 5,

                            max_features = 5,

                            n_estimators = 50,

                            criterion = "entropy",

                            random_state = SEED, 

                            n_jobs=-1)



start = time.time()

rf.fit(reshaped_trainX, trainy)



predicted = rf.predict(reshaped_testX)

elapsed_time = time.time() - start



rf_acc = accuracy_score(testy, predicted)

rf_time = elapsed_time



print("O treinamento levou %.0f segundos para os parâmetros padrão.\nAcurácia: %.4f" % (elapsed_time, rf_acc))
print ("Reporte de classificação:\n")

print(classification_report(testy, predicted))
plot_confusion_mtx2(rf, reshaped_testX, testy, 'Random Forest Basal')
space = {'max_depth': hp.quniform('max_depth', 1, 100, 1),

         'max_features': hp.quniform('max_features', 1, 50, 1),

         'n_estimators': hp.quniform('n_estimators', 25, 500, 5),

         'criterion': hp.choice('criterion', ["gini", "entropy"])}
def rf_tuning(space):

    

    global best_score, best_rf_model

    

    clf = RandomForestClassifier(max_depth = int(space['max_depth']),

                                 max_features = int(space['max_features']),

                                 n_estimators = int(space['n_estimators']), 

                                 criterion = space['criterion'], n_jobs=-1, random_state = SEED)

    

    clf.fit(reshaped_trainX, trainy)



    pred = clf.predict(reshaped_testX)

    accuracy = 1-accuracy_score(testy, pred)

    

    if (accuracy < best_score):

        best_score = accuracy

        best_rf_model = clf

    

    return {'loss': accuracy, 'status': STATUS_OK }
trials = Trials()

start = time.time()

neval = 50

best_score = 1.0

best_rf_model = []



best = fmin(fn = rf_tuning,

            space = space,

            algo = tpe.suggest,

            max_evals = neval,

            trials = trials,

            rstate = np.random.RandomState(SEED))



elapsed_time = time.time() - start



rf_optim_acc = (1-best_score)

rf_optim_time = elapsed_time
print("A otimização de parâmetros levou %.0f segundos para %d rodadas.\nAcurácia: %.4f\n\nParâmetros ótimos encontrado:\n%s" % (elapsed_time, neval, rf_optim_acc, best))
predicted = best_rf_model.predict(reshaped_testX)
print ("Reporte de classificação:\n")

print(classification_report(testy, predicted))
plot_confusion_mtx2(best_rf_model, reshaped_testX, testy, 'Random Forest Otimizado')
print("Comparação das acurácias\n\nRede Neural Simples:\t\t\t%.2f%%\nRede Neural Simples (não norm.):\t%.2f%%\nRede Neural Profunda:\t\t\t%.2f%%\nRandom Forest Basal:\t\t\t%.2f%%\nRandom Forest Otimizado:\t\t%.2f%%" % (NN_acc*100, NN_acc_ns*100, DNN_acc*100, rf_acc*100, rf_optim_acc*100))

print("\nComparação dos tempos\n\nRede Neural Simples:\t\t\t%.0f s\nRede Neural Simples (não norm.):\t%.0f s\nRede Neural Profunda:\t\t\t%.0f s\nRandom Forest Basal:\t\t\t%.0f s\nRandom Forest Otimizado:\t\t%.0f s" % (NN_time, NN_time_ns, DNN_time, rf_time, rf_optim_time))