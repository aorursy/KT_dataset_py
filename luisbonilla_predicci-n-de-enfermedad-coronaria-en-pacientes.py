# Librerias de Python 

import tensorflow as tf



#Librerias SKLearn 

from sklearn import decomposition, preprocessing, svm 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score



#Librerias Keras 

import keras

from keras.models import Sequential

from keras.layers import Dense, Input, Dropout

from keras import regularizers

from keras.callbacks import History 

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical



#otras librerias

import numpy as np

from __future__ import print_function

import pandas as pd

from pandas import read_excel

from itertools import cycle

from scipy import interp

from matplotlib import pyplot as plt

from scipy import stats



history = History()

%matplotlib inline
# Usar la libreria de Pandas para insertar la data 

pD = pd.read_csv("../input/data.csv",header = None, low_memory = False)



# Convierte los valores a una matriz N-Dimensional donde la N es el numero de rango de la matriz

pData = pD.as_matrix()
#convierte cada '?' de la data a un 'nan (not a number)' y luego convierte el array de strings a floats 

lab = pData[0,:];

pData = np.delete(pData, (0), axis=0)

pData[pData == '?'] = np.nan;

#transformacion a floats

pData = pData.astype('float32')





# Limpieza de datos

# Toma todas las columnas que no estén completamente vacías, concatena los datos de destino y elimina cualquier fila con los datos faltantes restantes.

d = pData[:,0:pData.shape[1]-4];

d = np.hstack((d,pData[:,pData.shape[1]-1].reshape(len(pData[:,0]),1)));

d = d[~np.isnan(d).any(axis=1)]

targets = d[:,d.shape[1]-1];

d = np.delete(d, (d.shape[1]-1), axis=1)

# uso de LDA (Linear Discriminant Analysis) and normalizacion min-max

dN = preprocessing.minmax_scale(d, feature_range=(-1, 1), axis=0, copy=True)

lda = LinearDiscriminantAnalysis(n_components=3)

X = lda.fit(dN, targets).transform(dN)  

targets = np.reshape(targets.astype(int),(len(targets),1))
#np.hstack combina las matrices NumPy juntas en la dirección "horizontal".

x = np.hstack((X,targets))

# sort para un ploteo mas sencillo

x = x[x[:,1].argsort()]
# Cuántos pacientes pertenecen a cada grupo para plotear de acuerdo.

type0 = sum(np.isin(x[:,1], 0));

type1 = sum(np.isin(x[:,1], 1));

q=0;

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(111)

ax1.bar(np.linspace(1,type0,type0), x[0:type0,q], align='center', label='Ninguno')

ax1.bar(np.linspace(type0+1,type0+type1,type1), x[type0:type0+type1,q], color='red', align='center', label='Afligidos')

plt.xlabel('Paciente',fontsize=18)

plt.ylabel('LDA Loading',fontsize=18)

plt.title('1D LDA Ploteo en Barra',fontsize=18)

plt.legend(loc='upper left',prop={'size': 18});

plt.show()
# arbol de decision y cross validation

clf = DecisionTreeClassifier(random_state=0)

fiveF = cross_val_score(clf, x[:,0].reshape(len(x[:,0]),1), x[:,1], cv=5)

print("Todos: ", fiveF, ". \nPromedio: ", np.mean(fiveF) )
X = x[:,0].reshape(x[:,0].shape[0],1);

# X = x[:,0];

y = x[:,1];

n_samples, n_features = X.shape

cv = StratifiedKFold(n_splits=5)

classifier = svm.SVC(kernel='linear', probability=True,

                     random_state=0)



tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(10,10))

i = 0

for train, test in cv.split(X, y):

    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

    # Computar curvatura ROC curve ay area de la curva

    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

    tprs.append(interp(mean_fpr, fpr, tpr))

    tprs[-1][0] = 0.0

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3,

             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))



    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

         label='Chance', alpha=.8)



mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',

         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

         lw=2, alpha=.8)



std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                 label=r'$\pm$ 1 std. dev.')



plt.xlim([-0.01, 1.01])

plt.ylim([-0.01, 1.01])

plt.xlabel('Tasa de falso positivo',fontsize=18)

plt.ylabel('Tasa de positivos verdaderos',fontsize=18)

plt.title('Cross-Validation ROC de LDA',fontsize=18)

plt.legend(loc="lower right", prop={'size': 15})

plt.show()
# min-max normalization

y = np_utils.to_categorical(targets,num_classes=2)

dN = preprocessing.minmax_scale(d, feature_range=(0, 1), axis=0, copy=True)

# 50/50 entrenamiento/prueba

train_X, test_X, train_y, test_y = train_test_split(dN, y, train_size=0.5, random_state=0)
# Definicion del modelo



def create_baseline():

    model = Sequential()

    model.add(Dense(15, activation='relu',input_shape=(10,)))

    model.add(Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

    model.add(Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001)))

    keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy',auc_roc])

    return model





def auc_roc(y_true, y_pred):

    # metrica de Tensorflow

    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)



    # buscar todas las variables creadas para esta métrica

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]



    # Agregar variables métricas a GLOBAL_VARIABLES.

    # Se inicializarán para una nueva sesión.

    for v in metric_vars:

        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)



    # forzar a actualizar valores de métrica

    with tf.control_dependencies([update_op]):

        value = tf.identity(value)

        return value
# Entrenamiento

model = create_baseline();

history = model.fit(train_X, train_y,

          validation_data=(test_X, test_y),

          batch_size=32, epochs=700, verbose=1)
# Curva AUROC y precisión de prueba para métrica de rendimiento

y_pred = model.predict_proba(test_X);

fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y[:,0], y_pred[:,0]);

auc_keras = auc(fpr_keras, tpr_keras);

accuracy = np.mean(np.equal(test_y, np.round(y_pred)));

plt.figure(figsize=(10,10))

plt.plot(fpr_keras, tpr_keras, color='black', label='AUC = {:.3f}'.format(auc_keras));

plt.xlabel('Tasa de falso positivo',fontsize=18);

plt.ylabel('Tasa de positivos verdaderos',fontsize=18);

plt.title('ROC curve: Max-Min Normalizado - Prueba de precisión = %0.2f' % (accuracy),fontsize=18);

plt.legend(loc='lower right',fontsize=18);



# entrenar y probar la perdida (Loss)

train_loss = history.history['loss']

val_loss   = history.history['val_loss']

xc         = range(700)

_=plt.figure(figsize=(10,10))

plt.plot(xc, train_loss,label='Entrenamiento')

plt.plot(xc, val_loss, label='Validacion')

plt.xlabel('Epochs',fontsize=18)

plt.ylabel('Perdida(Loss)',fontsize=18)

plt.title('Cost Curves',fontsize=18)

plt.legend(loc="upper right", prop={'size': 15})
#resultados

model = ExtraTreesClassifier();

model.fit(dN, y);

importance = pd.DataFrame({ '1. Parametros' : lab[0:-4], '2. Importancia' : model.feature_importances_});

importance