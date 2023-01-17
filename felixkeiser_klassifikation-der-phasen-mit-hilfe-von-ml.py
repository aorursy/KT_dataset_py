%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras import optimizers

from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization

from keras.models import *

from keras.callbacks import ModelCheckpoint

from keras. optimizers import *

from keras.utils.vis_utils import plot_model

import numpy as np

import pickle, os

import time

import warnings

from sklearn import linear_model

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from urllib.request import urlopen 
np.random.seed(42)





L=40 

J=-1.0 

T=np.linspace(0.25,4.0,16)

T_c=2.26
url_main = 'https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/';



data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 

label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"



data = pickle.load(urlopen(url_main + data_file_name)) # 1d bit array

data = np.unpackbits(data).reshape(-1, 1600) # dekompression der Bits

data=data.astype('int')

data[np.where(data==0)]=-1



labels = pickle.load(urlopen(url_main + label_file_name))
num_classes=2

train_to_test_ratio=0.5 # Verhältniss Trainingdaten/Gesamtdaten. Bei 0.5 gibt es also genauso viele Testdaten wie Trainingsdaten



X_ordered=data[:70000,:] # Die ersten 70000 Einträge gehöhren zu den Temp. 0.25 - 1.75 und sind somit geordnet

Y_ordered=labels[:70000]



X_critical=data[70000:100000,:]

Y_critical=labels[70000:100000]



X_disordered=data[100000:,:] # Ab 10000 gehöhren die Einträge gehöhren zu den Temp. > 2.5 und sind somit ungeordnet

Y_disordered=labels[100000:]



del data,labels



X=np.concatenate((X_ordered,X_disordered)) # Dies entspricht dem Datensatz ohne kritische Konfigurationen

Y=np.concatenate((Y_ordered,Y_disordered))



# Teilt in Trainings- und Testdaten auf

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio,test_size=1.0-train_to_test_ratio)



X=np.concatenate((X_critical,X))

Y=np.concatenate((Y_critical,Y))



print('X_train shape:', X_train.shape)

print('Y_train shape:', Y_train.shape)



print()

print(X_train.shape[0], 'train samples')

print(X_critical.shape[0], 'critical samples')

print(X_test.shape[0], 'test samples')
cmap_args=dict(cmap='plasma_r')



fig, axarr = plt.subplots(nrows=1, ncols=3)



axarr[0].imshow(X_ordered[20001].reshape(L,L),**cmap_args)

axarr[0].set_title('$\\mathrm{ordered\\ phase}$',fontsize=16)

axarr[0].tick_params(labelsize=16)



axarr[1].imshow(X_critical[10001].reshape(L,L),**cmap_args)

axarr[1].set_title('$\\mathrm{critical\\ region}$',fontsize=16)

axarr[1].tick_params(labelsize=16)



im=axarr[2].imshow(X_disordered[50001].reshape(L,L),**cmap_args)

axarr[2].set_title('$\\mathrm{disordered\\ phase}$',fontsize=16)

axarr[2].tick_params(labelsize=16)



fig.subplots_adjust(right=2.0)



plt.show()


lmbdas=np.logspace(-5,5,11)





train_accuracy=np.zeros(lmbdas.shape,np.float64)

test_accuracy=np.zeros(lmbdas.shape,np.float64)

critical_accuracy=np.zeros(lmbdas.shape,np.float64)



train_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)

test_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)

critical_accuracy_SGD=np.zeros(lmbdas.shape,np.float64)





for i,lmbda in enumerate(lmbdas):



    # Logistische Regression mit L2

    logreg=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                           solver='liblinear')



    logreg.fit(X_train, Y_train)



    train_accuracy[i]=logreg.score(X_train,Y_train)

    test_accuracy[i]=logreg.score(X_test,Y_test)

    critical_accuracy[i]=logreg.score(X_critical,Y_critical)

    

    

    # Sochastic Gradient Descent

    logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=lmbda, max_iter=100, 

                                           shuffle=True, random_state=1, learning_rate='optimal')



    logreg_SGD.fit(X_train,Y_train)



    train_accuracy_SGD[i]=logreg_SGD.score(X_train,Y_train)

    test_accuracy_SGD[i]=logreg_SGD.score(X_test,Y_test)

    critical_accuracy_SGD[i]=logreg_SGD.score(X_critical,Y_critical)

    
plt.semilogx(lmbdas,train_accuracy,'*-b',label='liblinear train')

plt.semilogx(lmbdas,test_accuracy,'*-r',label='liblinear test')

plt.semilogx(lmbdas,critical_accuracy,'*-g',label='liblinear critical')



plt.semilogx(lmbdas,train_accuracy_SGD,'*--b',label='SGD train')

plt.semilogx(lmbdas,test_accuracy_SGD,'*--r',label='SGD test')

plt.semilogx(lmbdas,critical_accuracy_SGD,'*--g',label='SGD critical')



plt.xlabel('$\\lambda$')

plt.ylabel('$\\mathrm{accuracy}$')



plt.grid()

plt.legend()





plt.show()
lmbdas=np.logspace(-5,5,11)



train_accuracy_l1=np.zeros(lmbdas.shape,np.float64)

test_accuracy_l1=np.zeros(lmbdas.shape,np.float64)

critical_accuracy_l1=np.zeros(lmbdas.shape,np.float64)



train_accuracy_el=np.zeros(lmbdas.shape,np.float64)

test_accuracy_el=np.zeros(lmbdas.shape,np.float64)

critical_accuracy_el=np.zeros(lmbdas.shape,np.float64)



for i,lmbda in enumerate(lmbdas):

    

    

    # Logistische Regression mit L1

    logreg_l1=linear_model.LogisticRegression(penalty='l1',C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,

                                              tol=1E-5,solver='liblinear')



    logreg_l1.fit(X_train, Y_train)



    train_accuracy_l1[i]=logreg_l1.score(X_train,Y_train)

    test_accuracy_l1[i]=logreg_l1.score(X_test,Y_test)

    critical_accuracy_l1[i]=logreg_l1.score(X_critical,Y_critical)

    

    

    # Logistische Regression mit L1 + L2 (elasticnet), dabei muss Solver=saga

    logreg_el=linear_model.LogisticRegression(penalty='elasticnet',C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                           solver='saga',l1_ratio=0.5)



    logreg_el.fit(X_train, Y_train)



    train_accuracy_el[i]=logreg_el.score(X_train,Y_train)

    test_accuracy_el[i]=logreg_el.score(X_test,Y_test)

    critical_accuracy_el[i]=logreg_el.score(X_critical,Y_critical)
plt.semilogx(lmbdas,train_accuracy_l1,'*-b',label='liblinear(l1) train')

plt.semilogx(lmbdas,test_accuracy_l1,'*-r',label='liblinear(l1) test')

plt.semilogx(lmbdas,critical_accuracy_l1,'*-g',label='liblinear(l1) critical')



plt.semilogx(lmbdas,train_accuracy_el,'*--b',label='Saga(l1+l2) train')

plt.semilogx(lmbdas,test_accuracy_el,'*--r',label='Saga(l1+l2) test')

plt.semilogx(lmbdas,critical_accuracy_el,'*--g',label='Saga(l1+l2) critical')



plt.xlabel('$\\lambda$')

plt.ylabel('$\\mathrm{accuracy}$')



plt.grid()

plt.legend()





plt.show()
lmbdas=np.logspace(-5,5,11)



train_accuracy_cg=np.zeros(lmbdas.shape,np.float64)

test_accuracy_cg=np.zeros(lmbdas.shape,np.float64)

critical_accuracy_cg=np.zeros(lmbdas.shape,np.float64)



train_accuracy_fgs=np.zeros(lmbdas.shape,np.float64)

test_accuracy_fgs=np.zeros(lmbdas.shape,np.float64)

critical_accuracy_fgs=np.zeros(lmbdas.shape,np.float64)





for i,lmbda in enumerate(lmbdas):

    

    

    # Logistische Regression mit newton-cg

    logreg_cg=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,

                                              tol=1E-5,solver='newton-cg')



    logreg_cg.fit(X_train, Y_train)



    train_accuracy_cg[i]=logreg_l1.score(X_train,Y_train)

    test_accuracy_cg[i]=logreg_l1.score(X_test,Y_test)

    critical_accuracy_cg[i]=logreg_l1.score(X_critical,Y_critical)

    

    

    # Logistische Regression mit lbgfs

    logreg_fgs=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                           solver='lbfgs')



    logreg_fgs.fit(X_train, Y_train)



    train_accuracy_fgs[i]=logreg_fgs.score(X_train,Y_train)

    test_accuracy_fgs[i]=logreg_fgs.score(X_test,Y_test)

    critical_accuracy_fgs[i]=logreg_fgs.score(X_critical,Y_critical)
plt.semilogx(lmbdas,train_accuracy_cg,'*-b',label='Newton-cg train')

plt.semilogx(lmbdas,test_accuracy_cg,'*-r',label='Newton-cg test')

plt.semilogx(lmbdas,critical_accuracy_cg,'*-g',label='Newton-cg critical')



plt.semilogx(lmbdas,train_accuracy_fgs,'*--b',label='lbgfs train')

plt.semilogx(lmbdas,test_accuracy_fgs,'*--r',label='lbgfs test')

plt.semilogx(lmbdas,critical_accuracy_fgs,'*--g',label='lbgfs critical')





plt.xlabel('$\\lambda$')

plt.ylabel('$\\mathrm{accuracy}$')



plt.grid()

plt.legend(loc='lower left')





plt.show()
X_supercritical=X_critical[20000:300000,:] # Es werden die Konfigurationen mit T = 2.5 ausgewählt

Y_supercritical=Y_critical[20000:300000]



lmbda_fgs = 1000

lmbda = 10000



logreg_fgs=linear_model.LogisticRegression(C=1.0/lmbda_fgs,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                               solver='lbfgs')

logreg_fgs.fit(X_train, Y_train)

crit_accuracy_fgs=logreg_fgs.score(X_supercritical,Y_supercritical)

print('Critical Accuracy with lbfgs:' + str(crit_accuracy_fgs))

    

    

logreg=linear_model.LogisticRegression(C=1.0/lmbda,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                               solver='liblinear')

logreg.fit(X_train, Y_train)

crit_accuracy=logreg.score(X_supercritical,Y_supercritical)

print('Critical Accuracy with liblinear:' + str(crit_accuracy))

    
X_own_2d = np.load("/kaggle/input/phasen-bergang-am-2-d-ising-modell/ml_eigene_daten_bilder.npy")

J = np.load('/kaggle/input/phasen-bergang-am-2-d-ising-modell/ml_eigene_daten_labels.npy')



X_own = X_own_2d.reshape(16000, 1600)

Y_own = np.where(J > 0.441, 1, 0)



not_crit = np.concatenate([np.where(J > 0.5)[0], np.where(J <= 0.4)[0]])

crit = np.setdiff1d(np.arange(16000), not_crit)



X_own_notcrit = X_own[not_crit]

Y_own_notcrit = Y_own[not_crit]

J_own_notcrit = J[not_crit]



X_own_crit = X_own[crit]

Y_own_crit = Y_own[crit]

J_own_crit = J[crit]



print("Noncritical Samples: " + str(len(Y_own_notcrit)))

print("Critical Samples: " + str(len(Y_own_crit)))
lmbda_fgs = 1000

lmbda_lib = 10000



logreg_fgs=linear_model.LogisticRegression(C=1.0/lmbda_fgs,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                               solver='lbfgs')

logreg_fgs.fit(X_train, Y_train)

crit_accuracy_fgs=logreg_fgs.score(X_own_crit,Y_own_crit)

notcrit_accuracy_fgs=logreg_fgs.score(X_own_notcrit,Y_own_notcrit)



logreg=linear_model.LogisticRegression(C=1.0/lmbda_lib,random_state=1,verbose=0,max_iter=1E3,tol=1E-5,

                                               solver='liblinear')

logreg.fit(X_train, Y_train)

crit_accuracy=logreg.score(X_supercritical,Y_supercritical)

notcrit_accuracy=logreg.score(X_own_notcrit,Y_own_notcrit)



print('Critical Accuracy with liblinear:' + str(crit_accuracy))

print('Non-Critical Accuracy with liblinear:' + str(notcrit_accuracy))

print('Critical Accuracy with lbfgs:' + str(crit_accuracy_fgs))

print('Non-Critical Accuracy with lbfgs:' + str(notcrit_accuracy_fgs))
# Silly Reshape

X_train, X_test, X_critical = X_train.reshape((-1, 40, 40, 1)), X_test.reshape((-1, 40, 40, 1)), X_critical.reshape((-1, 40, 40, 1))

X_own_notcrit, X_own_crit = X_own_notcrit.reshape((-1, 40, 40, 1)), X_own_crit.reshape((-1, 40, 40, 1))
in_layer = Input(shape=(40, 40, 1 ))



conv = Conv2D(10, (3,3), activation='relu', padding='same')(in_layer)

conv = BatchNormalization()(conv)

conv = Conv2D(20, (5,5), activation='relu', padding='same')(conv)

conv = BatchNormalization()(conv)

conv = MaxPooling2D((2,2))(conv)



conv = Conv2D(16, (3,3), activation='relu', padding='same')(conv)

conv = BatchNormalization()(conv)

conv = Conv2D(32, (4,4), activation='relu', padding='same')(conv)

conv = BatchNormalization()(conv)

conv = MaxPooling2D((2,2))(conv)



conv = Conv2D(16, (3,3), activation='relu', padding='same')(conv)

conv = BatchNormalization()(conv)

conv = Conv2D(32, (2,2), activation='relu', padding='same')(conv)

conv = BatchNormalization()(conv)



flat = Flatten()(conv)



dense = Dense(350, activation='relu')(flat)

dense = Dropout(0.9)(dense)

dense = Dense(50, activation='relu')(dense)

dense = Dropout(0.85)(dense)

out = Dense(1, activation='sigmoid')(dense)



model = Model(inputs=in_layer, outputs=out)



model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])



# plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=False, dpi=100)
filepath=f"{time.time()}-best_weights.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')



history = model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_data=[X_critical, Y_critical], callbacks=[checkpoint], verbose=0)
# Plot training & validation accuracy values

plt.figure(dpi=200, figsize=(20,5))

plt.plot(history.history['accuracy'], label='Trainings Daten')

plt.plot(history.history['val_accuracy'], label='Kritische Daten')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()



# Plot training & validation loss values

plt.figure(dpi=200, figsize=(20,5))

plt.plot(history.history['loss'], label='Trainings Daten')

plt.plot(history.history['val_loss'], label='Kritische Daten')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc=('upper right'))

plt.show()
model.load_weights(filepath)

scores_train = model.evaluate(X_train, Y_train, verbose=0)

print("Train Accuracy: %.2f%%" % (scores_train[1]*100))

scores_test = model.evaluate(X_test, Y_test, verbose=0)

print("Test Accuracy: %.2f%%" % (scores_test[1]*100))

scores_crit = model.evaluate(X_critical, Y_critical, verbose=0)

print("Critical Accuracy: %.2f%%" % (scores_crit[1]*100))

scores_crit = model.evaluate(X_own_notcrit, Y_own_notcrit, verbose=0)

print("Test Accuracy (own): %.2f%%" % (scores_crit[1]*100))

scores_crit = model.evaluate(X_own_crit, Y_own_crit, verbose=0)

print("Critical Accuracy (own): %.2f%%" % (scores_crit[1]*100))