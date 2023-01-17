import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from sklearn.metrics import mean_squared_error

import warnings

import sys

import itertools

from sklearn.metrics import classification_report





# Pacotes de Manipulação de Dados

import sklearn 

from sklearn.metrics import confusion_matrix, recall_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale





# Keras e TensorFlow

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop

import tensorflow as tf



# Pacotes para Confusion Matrix e Balanceamento de Classes

import imblearn



LABELS = ["Noraml", "Com Diabetes"]



%matplotlib inline
# Carregando a base de dados

df  = pd.read_csv('../input/dataset_treino.csv', decimal=b',')

# Eliminando a coluna id

df.drop(['id'], axis = 1, inplace = True)
# Verificando tipos de dados e estatísticas descritivas

print (df.info ()) 

print (df.describe ())
# Visualizando as primeiras 10 linhas do dataframe

df.head(10)
count_classes = pd.value_counts(df['classe'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Distribuição")

plt.xticks(range(2), LABELS)

plt.xlabel("Classe")

plt.ylabel("Frequência");
print('O Dataframe possui ' + str(df.shape[0]) + ' linhas e ' + str(df.shape[1]) + ' colunas')
diabetes = df.loc[df['classe'] == 1]

sem_diabetes = df.loc[df['classe'] == 0]

print("Temos", len(diabetes), "pontos de dados como diabetes e", len(sem_diabetes), "pontos de dados considerados normais.")
# Atribuindo Valores às Variáveis X e Y do Modelo

X = df.iloc[:,:-1]

y = df['classe']





# Aplicando Scala 

X = scale(X)



# Gerando dados de treino, teste e validação

X1, X_valid, y1, y_valid = train_test_split(X, y, test_size = 0.10, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.23, random_state = 0)
print("Tamanho do Dataset de Treino: ", X_train.shape)
print("Tamanho do Dataset de Validaçao: ", X_valid.shape)
print("Tamanho do Dataset de Test: ", X_test.shape)
from imblearn.under_sampling import RandomUnderSampler

uds = RandomUnderSampler(random_state=42)

X2_train, y2_train = uds.fit_sample(X_train, y_train)
count_classes = pd.value_counts(y2_train, sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Distribuição")

plt.xticks(range(2), LABELS)

plt.xlabel("Classe")

plt.ylabel("Frequência");
from keras.callbacks import EarlyStopping

from tensorflow import set_random_seed

import keras as keras

from sklearn.metrics import precision_score, recall_score
# Função para construir a Confusion Metrix

def pretty_print_conf_matrix(y_true, y_pred, 

                             classes,

                             normalize=False,

                             title='Confusion matrix',

                             cmap=plt.cm.Blues):

    """

    referência: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py



    """



    cm = confusion_matrix(y_true, y_pred)



    # Configure Confusion Matrix Plot Aesthetics (no text yet) 

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(cm, cmap=plt.cm.Blues)

    fig.colorbar(cax)

    plt.title(title, fontsize=14)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    plt.ylabel('True label', fontsize=12)

    plt.xlabel('Predicted label', fontsize=12)



    # Calculate normalized values (so all cells sum to 1) if desired

    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]



    # Place Numbers as Text on Confusion Matrix Plot

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black",

                 fontsize=12)





    # Add Precision, Recall, F-1 Score as Captions Below Plot

    rpt = classification_report(y_true, y_pred)

    rpt = rpt.replace('avg / total', '      avg')

    rpt = rpt.replace('support', 'N Obs')



    plt.annotate(rpt, 

                 xy = (0,0), 

                 xytext = (-50, -140), 

                 xycoords='axes fraction', textcoords='offset points',

                 fontsize=12, ha='left')    



    # Plot

    plt.tight_layout()
# Função para as estatísticas de precisão, imprecisão, falsos negativos e falsos positivos

def estatisticas(y_true, y_pred):

    false_neg = 0

    false_pos = 0

    incorrect = 0

    y2_true = np.array(y_true)

    total = len(y_true)

    for i in range(len(y_true)):        

        if y_pred[i] != y2_true[i]:

            incorrect += 1

            if y2_true[i] == 1 and y_pred[i] == 0:

                false_neg += 1

            else:

                false_pos += 1



    inaccuracy = incorrect / total



    print('Inacurácia:', inaccuracy)

    print('Acurácia:', 1 - inaccuracy)

    if incorrect > 0:

        print('Taxa de Falsos Negativos:', false_neg/incorrect)

        print('Taxa de Falsos Positivos:', false_pos / incorrect )    

    print('Falsos Negativos/total:', false_neg/total)

    return inaccuracy, incorrect
# Classe para calcular a métrica de precisão com base no recall

class Metrics(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self._data = []



    def on_epoch_end(self, batch, logs={}):

        X_val, y_val = self.validation_data[0], self.validation_data[1]

        y_predict = np.round(model2.predict(X_val)).T[0]

    

        self._data.append({

            'val_recall': recall_score(y_val, np.round(model2.predict(X_val)).T[0], pos_label = 1, average = 'binary'),

            'val_precision': precision_score(y_val, np.round(model2.predict(X_val)).T[0],  pos_label = 0 , average = 'binary'),

        })

        return



    def get_data(self):

        return self._data
# Parametrtos da rede

batch_size = 10

seed = 7

set_random_seed(seed)

metrics = Metrics()



OPTIMIZER = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.001)
model2 = Sequential()

model2.add(Dense(16, input_dim = 8, kernel_initializer='normal')) 

model2.add(Activation('relu'))

model2.add(Dropout(0.20))

model2.add(Dense(8,  kernel_initializer='uniform'))

model2.add(Activation('tanh'))

model2.add(Dense(1,  kernel_initializer='uniform'))

model2.add(Activation('sigmoid'))

monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 10, verbose = 1, mode = 'auto')   

model2.compile(loss = 'binary_crossentropy', optimizer = OPTIMIZER, metrics = ['accuracy'])

model2.summary()
history = model2.fit(X2_train, y2_train, epochs = 42, batch_size = batch_size, validation_data=(X_valid, y_valid), callbacks = [metrics], shuffle=False)
# Avaliando o modelo

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model train vs validation loss'), 

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
print("Loss: ", model2.evaluate(X_valid, y_valid, verbose=0))
from sklearn import metrics

probs = model2.predict_proba(X_valid)

preds = probs[:,0]

fpr, tpr, threshold = metrics.roc_curve(y_valid, preds)

roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True  Positive rate')

plt.xlabel('False Positive rate')

plt.show()
y2_predicted = np.round(model2.predict(X_test)).T[0]

y2_correct = y_test
np.setdiff1d(y2_predicted, y2_correct)
inaccuracy, incorrect = estatisticas(y2_correct, y2_predicted)
print('Validation Results')

print(recall_score(y_valid,np.round(model2.predict(X_valid)).T[0]))

print('\nTest Results')

print(1 - inaccuracy)

print(recall_score(y_test,np.round(model2.predict(X_test)).T[0]))
print(incorrect)
# Plot Confusion Matrix

warnings.filterwarnings('ignore')

pretty_print_conf_matrix(y2_correct, y2_predicted, 

                         classes= ['0', '1'],

                         title='Confusion Matrix')
# Carregando o dataset de teste

dfteste = pd.read_csv('../input/dataset_teste.csv', decimal=b',')

dft = dfteste.iloc[:,1:]
# Aplicando Scala

X_test2 = scale(dft)
# Fazendo as previsões

y_test2 = np.round(model2.predict(X_test2))

dfteste['classe'] = y_test2.astype(np.int64)
# Eliminando as colunas para gerar o sampleSubmission.csv

dfteste.drop(['num_gestacoes'], axis = 1, inplace = True)

dfteste.drop(['glicose'], axis = 1, inplace = True)

dfteste.drop(['pressao_sanguinea'], axis = 1, inplace = True)

dfteste.drop(['grossura_pele'], axis = 1, inplace = True)

dfteste.drop(['insulina'], axis = 1, inplace = True)

dfteste.drop(['bmi'], axis = 1, inplace = True)

dfteste.drop(['indice_historico'], axis = 1, inplace = True)

dfteste.drop(['idade'], axis = 1, inplace = True)



print(dfteste)
# Salvando 

dfteste.to_csv('./sampleSubmission.csv', index=False)