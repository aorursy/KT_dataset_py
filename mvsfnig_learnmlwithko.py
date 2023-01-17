''' basic libs '''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import warnings



''' network '''

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop, Adam



''' onde hot ecoding'''

from keras.utils import to_categorical



''' curva ROC '''

from sklearn.metrics import roc_curve, auc



''' configure  '''

%matplotlib inline

warnings.filterwarnings('ignore')

plt.style.use('seaborn')



print(os.listdir("../input"))
%time df = pd.read_csv('../input/train.csv')

df.head()
df.describe()
''' select features ans labels '''

features = df.ix[:,1:257]

labels = df.ix[:,257]
print('SHAPE -> Features :',features.shape ,'Labels :',labels.shape, 'df Rows :', len(df), 'df Columms :', len(df.columns))
''' selecionado uma amostra '''



size = 262144



''' convert data in tensor '''

x = np.array(features.ix[:size,:])

y = np.array(labels.ix[:size])



''' tranform data in float '''

x = x.astype('float')



''' normalize your data '''

x -= x.mean(axis=0)

x /= x.std(axis=0)



''' one hot encoding '''

y = to_categorical(y)



print(x.shape, y.shape)
def proportion_split(total, verbose=False):

    p_train = int(total/2)

    p_test  = int(p_train/2)

    p_lim_test = (p_train+p_test)

    if verbose:

        print('Train(:%i) Test(%i:%i) Validation(:%i)' %(p_train,p_train,p_lim_test,p_lim_test))

        print('-'*30)

        return p_train, p_lim_test

    else:

        return p_train, p_lim_test



ini, end = proportion_split(size, verbose=True)



x_train = x[:ini,]

x_test = x[ini:end,]

x_val = x[end:,]



y_train = y[:ini,]

y_test = y[ini:end,]

y_val = y[end:,]



print('x_train'+str(x_train.shape)+' y_train'+str(y_train.shape)+

      ' | x_test'+str(x_test.shape)+' y_test'+str(y_test.shape)+

      ' | x_val'+str(x_val.shape)+' y_val'+str(y_val.shape))

num_classes = 2

epocas = 20

lote = 32

taxa_aprendizado = 0.001
def NeuralNetwork(entrada):

    model = Sequential()

    model.add(Dense(16, activation='relu', input_shape=(entrada,)))

    model.add(Dropout(0.2))

    model.add(Dense(16, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='sigmoid'))

    

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model
nn = NeuralNetwork(x.shape[1])
%time hist = nn.fit(x_train,y_train, epochs=20, validation_data=(x_val, y_val), verbose=0)
def print_score(metricas, evaluation):

    print('_'*20)

    print('Model Evaluate')

    print('-'*20)

    for i in range(len(evaluation)):

        print(metricas[i]+' = %.2f' %(evaluation[i]))

    print('-'*20)

        

def plot_log_train(log):

    chaves = list(log.keys())

    print(chaves)

    plt.figure(figsize=(15,6))

    for i in range(len(chaves)):

        plt.plot(log[chaves[i]], '-o', label=chaves[i])

    plt.legend()

    plt.show()

    

def plots_log_train(log):

    chaves = list(log.keys())

    fig = plt.figure(figsize=(18,5))

    

    ax = fig.add_subplot(121)

    ax.plot(log[chaves[0]], '-o', label=chaves[0])

    ax.plot(log[chaves[2]], '-o', label=chaves[2])

    ax.set_title('Loss')

    ax.legend()

    

    ax = fig.add_subplot(122)

    ax.set_title('Accuracy')

    ax.plot(log[chaves[1]], '-o', label=chaves[1])

    ax.plot(log[chaves[3]], '-o', label=chaves[3])

    ax.legend()

    

    plt.show()
plots_log_train(hist.history)

res = nn.evaluate(x_test, y_test)

print_score(nn.metrics_names, res)
y_pred = nn.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())

auc = auc(fpr, tpr)
''' plot Curve ROC '''

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='NN (area = {:.3f})'.format(auc))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()