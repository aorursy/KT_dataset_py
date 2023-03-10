import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt



from keras.wrappers.scikit_learn import KerasClassifier

from keras import optimizers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("../input/pre-processed-db/pre-processed-base(train)", sep="\t")



#removendo colunas que nao existem no conjunto de validacao

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_0', axis=1, inplace=True)

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_1', axis=1, inplace=True)

df_train.drop('CLASSE_SOCIAL_PCNC_PCM_rv_2', axis=1, inplace=True)



#transformando coluna categorica em numerica

df_train['PROPHET_NORM_FEATURES'] = df_train['PROPHET_NORM_FEATURES'].astype('category')

df_train['PROPHET_NORM_FEATURES'] = df_train['PROPHET_NORM_FEATURES'].cat.codes

#df_train.drop('PROPHET_NORM_FEATURES', axis=1, inplace=True)



#-.-

df_train.drop('PROPHET_LABEL', axis=1, inplace=True)

df_train.drop('NEURO_LABEL', axis=1, inplace=True)
df_valid = pd.read_csv("../input/pre-processed-db/pre-processed-base(valid)", sep="\t")



#transformando coluna categorica em numerica

df_valid['PROPHET_NORM_FEATURES'] = df_valid['PROPHET_NORM_FEATURES'].astype('category')

df_valid['PROPHET_NORM_FEATURES'] = df_valid['PROPHET_NORM_FEATURES'].cat.codes

#df_train.drop('PROPHET_NORM_FEATURES', axis=1, inplace=True)



#-.-

df_valid.drop('PROPHET_LABEL', axis=1, inplace=True)

df_valid.drop('NEURO_LABEL', axis=1, inplace=True)
y_train = df_train.iloc[:,0].values

X_train = df_train.iloc[:,1:].values



y_valid = df_valid.iloc[:,0].values

X_valid = df_valid.iloc[:,1:].values



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_valid = scaler.fit_transform(X_valid)
def extract_final_losses(history):

    """Fun????o para extrair o melhor loss de treino e valida????o.

    

    Argumento(s):

    history -- Objeto retornado pela fun????o fit do keras.

    

    Retorno:

    Dicion??rio contendo o melhor loss de treino e de valida????o baseado 

    no menor loss de valida????o.

    """

    train_loss = history.history['loss']

    val_loss = history.history['val_loss']

    idx_min_val_loss = np.argmin(val_loss)

    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}



def plot_training_error_curves(history):

    """Fun????o para plotar as curvas de erro do treinamento da rede neural.

    

    Argumento(s):

    history -- Objeto retornado pela fun????o fit do keras.

    

    Retorno:

    A fun????o gera o gr??fico do treino da rede e retorna None.

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
from sklearn.ensemble import GradientBoostingClassifier



#variando n_estimators (segundo a documenta????o valores maiores s??o melhores), max_depth com valor padr??o

gb_clf = GradientBoostingClassifier(learning_rate=0.4, n_estimators=150, max_depth=3, verbose=10)

gb_clf.fit(X_train, y_train)

gb_pred_class = gb_clf.predict(X_valid)

gb_pred_scores = gb_clf.predict_proba(X_valid)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_valid, gb_pred_class, gb_pred_scores)

print('Performance no conjunto de valida????o:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

print('Matriz de confus??o no conjunto de valida????o:')

print(confusion_matrix(y_valid, gb_pred_class))