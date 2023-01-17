# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras import optimizers

from keras.wrappers.scikit_learn import KerasClassifier



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/BASE-PREPROCESSED(TRAIN)", sep="\t")



#removendo colunas que nao existem no conjunto de validacao

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_0', axis=1, inplace=True)

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_1', axis=1, inplace=True)

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_2', axis=1, inplace=True)



#removendo coluna categorica

df_train.drop('PROPHET_NORM_FEATURES', axis=1, inplace=True)



#-.-

df_train.drop('PROPHET_LABEL', axis=1, inplace=True)

df_train.drop('NEURO_LABEL', axis=1, inplace=True)



df_train.head()
df_valid = pd.read_csv("../input/BASE-PREPROCESSED(VALIDACAO)", sep="\t")



#removendo coluna categorica

df_valid.drop('PROPHET_NORM_FEATURES', axis=1, inplace=True)



#-.-

df_valid.drop('PROPHET_LABEL', axis=1, inplace=True)

df_valid.drop('NEURO_LABEL', axis=1, inplace=True)



df_valid.head()
y_train = df_train.iloc[:,0].values

X_train = df_train.iloc[:,1:].values



y_valid = df_valid.iloc[:,0].values

X_valid = df_valid.iloc[:,1:].values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_valid = scaler.fit_transform(X_valid)
def extract_final_losses(history):

    """Função para extrair o melhor loss de treino e validação.

    

    Argumento(s):

    history -- Objeto retornado pela função fit do keras.

    

    Retorno:

    Dicionário contendo o melhor loss de treino e de validação baseado 

    no menor loss de validação.

    """

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    idx_min_val_loss = np.argmin(val_loss)

    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}



def plot_training_error_curves(history):

    """Função para plotar as curvas de erro do treinamento da rede neural.

    

    Argumento(s):

    history -- Objeto retornado pela função fit do keras.

    

    Retorno:

    A função gera o gráfico do treino da rede e retorna None.

    """

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    fig, ax = plt.subplots()

    ax.plot(train_loss, label='Train')

    ax.plot(val_loss, label='Validation')

    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')

    ax.legend()

    plt.show()



def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):

    accuracy = accuracy_score(y, y_pred_class)

    recall = recall_score(y, y_pred_class, average='micro')

    precision = precision_score(y, y_pred_class, average='micro')

    f1 = f1_score(y, y_pred_class, average='micro')

    performance_metrics = (accuracy, recall, precision, f1)

    if y_pred_scores is not None:

        skplt.metrics.plot_ks_statistic(y, y_pred_scores)

        plt.show()

        y_pred_scores = y_pred_scores[:, 1]

        auroc = roc_auc_score(y, y_pred_scores)

        aupr = average_precision_score(y, y_pred_scores)

        performance_metrics = performance_metrics + (auroc, aupr)

    return performance_metrics



def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):

    print()

    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))

    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))

    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))

    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))

    if auroc is not None:

        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))

    if aupr is not None:

        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
input_dim = X_train.shape[1]



def create_sklearn_compatible_model():

    model = Sequential()

    model.add(Dense(100, activation='tanh', input_dim=input_dim))

    model.add(Dense(1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=3.0, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='mean_squared_error')

    return model
mlp_ens_clf = KerasClassifier(build_fn=create_sklearn_compatible_model,

                              batch_size=64, epochs=50, verbose=0)

ens_clf = VotingClassifier([('mlp', mlp_ens_clf)], 

                           voting='soft')



ens_clf.fit(X_train, y_train)

ens_pred_class = ens_clf.predict(X_valid)

ens_pred_scores = ens_clf.predict_proba(X_valid)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_valid, ens_pred_class, ens_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)