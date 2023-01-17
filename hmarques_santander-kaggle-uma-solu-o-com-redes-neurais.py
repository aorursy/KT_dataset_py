import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.initializers import TruncatedNormal, RandomUniform, RandomNormal
from keras.constraints import unit_norm, max_norm

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, \
            recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print ('\nRealizando carga de dados....\n')
X_train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv", index_col='ID_code')
X_test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv", index_col='ID_code')
print ('\nCarga de dados concluída....\n')


print ('\nPreparando os conjuntos de treinamento e teste....\n')
y_train = X_train.target
X_train.drop(['target'], axis=1, inplace=True)
X_test_index = X_test.index

X_train.head()
X_test.head()
sum(X_train.columns.isnull())
sum(X_test.columns.isnull())
sns.set()
plt.figure(figsize=(30,3))
pos = 0
for col in X_train.columns.values[0:9]:
        pos+=1
        plt.subplot(190+pos)
        plt.hist(X_train[col], density=True, bins=60)
        plt.title(col)
        plt.ylabel('Probability')
        plt.xlabel('Data')
print ('\nGráficos das primeiras colunas...\n')
plt.show()
std_threshold = 8
bins = 6
print ('\nEstabelecendo as metricas para criação e features...\n')

    
#n_unicos_train = X_train.nunique()
#n_unicos_test = X_test.nunique()
    
std_train = X_train.std(axis=0)
std_test = X_test.std(axis=0)
    
col_train_interested = np.where(std_train >= std_threshold)
col_test_interested = np.where(std_test >= std_threshold)
    
col_train_for_bins = np.where(std_train < std_threshold)
col_test_for_bins = np.where(std_test < std_threshold)
print ('\nDiscricionando colunas com valores pequenos de Desvio Padrão...\n')
     
X_train_e = X_train[std_train.index[(col_train_for_bins)]]
X_test_e = X_test[std_test.index[(col_test_for_bins)]]
    
discretizer = preprocessing.KBinsDiscretizer(n_bins=bins, encode='onehot', strategy='uniform')
    
discretizer.fit(X_train_e)
sparse_matrix = discretizer.transform(X_train_e)
train_onehot = sparse_matrix.todense()

discretizer.fit(X_test_e)
sparse_matrix = discretizer.transform(X_test_e)
test_onehot = sparse_matrix.todense()


print ('\nEliminando colunas com std < std_threshold ...\n')
    
X_train = X_train[std_train.index[(col_train_interested)]]
X_test = X_test[std_test.index[(col_test_interested)]]
X_train.head()
# Número de características com valores de desvio padrão maiores que std_threshold
X_train.shape
scaler =  preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train
print ('\nConcatenando as matrizes...\n')   
X_train = np.concatenate((X_train, train_onehot), axis=1)
X_test = np.concatenate((X_test, test_onehot), axis=1)
X_train.shape

input_len = X_train.shape[1]


    
model = Sequential([
#Dense(input_len, input_shape=(input_len,)),
Dense(256, input_shape=(input_len,)),
Activation('relu'),
Dropout(0.5),
Dense(128, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.5),
Dense(64, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.3),
Dense(32, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.1),
Dense(1),
Activation('sigmoid'),
])
    
    
opt = optimizers.Adam(learning_rate=0.00005)    
#opt = optimizers.Adam(learning_rate=0.001)
    
        
model.compile(optimizer=opt,
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
    
print(model.summary())


history = model.fit(X_train, y_train, \
                epochs= 120,\
                batch_size=1024,\
                validation_split=0.1,\
                )
           
sns.set()
plt.figure(figsize=(15, 7))

plt.subplot(141)
plt.plot(history.history['accuracy'], label='ACC')
plt.plot(history.history['val_accuracy'], label='Val_ACC')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()


plt.subplot(142)
plt.plot(history.history['loss'], label='LOSS')
plt.plot(history.history['val_loss'], label='Val_LOSS')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
input_len = X_train.shape[1]


model = Sequential([
#Dense(input_len, input_shape=(input_len,)),
Dense(256, input_shape=(input_len,)),
Activation('relu'),
Dropout(0.5),
Dense(128, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.5),
Dense(64, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.3),
Dense(32, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.1),
Dense(1),
Activation('sigmoid'),
])
    
    
opt = optimizers.Adam(learning_rate=0.00005)    
#opt = optimizers.Adam(learning_rate=0.001)
    
        
model.compile(optimizer=opt,
             loss= 'binary_crossentropy',
              metrics=['accuracy'])
    
print(model.summary())


history = model.fit(X_train, y_train, \
                epochs= 40,\
                batch_size=1024,\
                #validation_split=0.1,\
                #class_weight = compute_class_weight('balanced', y_train.sum(), len(y_train))
                )
X_test.shape
predictions = model.predict(X_test)
   
pred = predictions.reshape((200000,))
output = pd.DataFrame({'ID_code': X_test_index, 'target': pred})
output.to_csv('ANN_Santander_GColab_fet_Eng_TPU_3.csv', index=False)
print("O arquivo para envio ao Kaggle foi salvo com sucesso!")
predictions[0:10]