import numpy as np
import pandas as pd 

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

!ls '../input/'
train = pd.read_csv('../input/train1.csv', index_col=0)
val = pd.read_csv('../input/val1.csv', index_col ='INDEX')
test = pd.read_csv('../input/test.csv',index_col = 'INDEX')
cols_train = [col for col in train.columns if col != 'y']
cols_test = [col for col in test.columns if col != 'y']
X_train = train[cols_train]
y_train = train['y']
X_test = test[cols_test]
y_test = test['y']
X_val = val[cols_test]
y_val = val['y']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
input_dim = X_train.shape[1]

mlp = Sequential()

mlp.add(Dense(8,activation='relu',input_dim = input_dim))
mlp.add(Dense(8,activation='relu'))
mlp.add(Dense(8,activation='relu'))
mlp.add(Dense(8,activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))
mlp.compile(optimizer='adam',loss='mean_squared_error')

history = mlp.fit(X_train, y_train, batch_size=64, epochs= 10000,callbacks=[EarlyStopping(patience=100)], validation_data=[X_val,y_val])
def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        ks = ks_2samp(y, y_pred_scores)[0]
        performance_metrics = performance_metrics + (auroc, aupr, ks)
    return performance_metrics

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

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None, ks=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
    if ks is not None:
        print("{metric:<18}{value:.4f}".format(metric="KS:", value=ks))
y_pred_scores = mlp.predict(X_test)
y_pred_class = mlp.predict_classes(X_test)

## Matriz de confusão
print('Matriz de confusão no conjunto de teste:')
print(confusion_matrix(y_test, y_pred_class))

## Resumo dos resultados
losses = extract_final_losses(history)
print()
print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr, ks = compute_performance_metrics(y_test, y_pred_class, y_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr, ks)
gdb = GradientBoostingClassifier()
rf = RandomForestClassifier()
gdb.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_gdb = gdb.predict(X_test)
y_pred_rf = rf.predict(X_test)

y_pred_scores_gdb = gdb.predict_proba(X_test)[:, 1]
y_pred_scores_rf = rf.predict_proba(X_test)[:, 1]

print('----- GRADIENT BOOSTING------')

print('Matriz de confusão no conjunto de teste:')
print(confusion_matrix(y_test, y_pred_gdb))

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr, ks = compute_performance_metrics(y_test, y_pred_gdb, y_pred_scores_gdb)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr, ks)

print('----- RANDOM FOREST ------')

print('Matriz de confusão no conjunto de teste:')
print(confusion_matrix(y_test, y_pred_rf))

print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr, ks = compute_performance_metrics(y_test, y_pred_rf, y_pred_scores_rf)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr, ks)