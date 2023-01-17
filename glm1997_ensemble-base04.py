import numpy as np

import pandas as pd



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from sklearn.metrics import roc_curve



from sklearn.preprocessing import MinMaxScaler



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt



from itertools import groupby



import pickle



import os
# print(os.listdir('../input/base04'))



#TREINO

pickle_off = open('../input/base04/X_train4s.pickle', 'rb')

X_train = pickle.load(pickle_off)



pickle_off = open('../input/base04/y_train4s.pickle', 'rb')

y_train = pickle.load(pickle_off)



#VALIDAÇÃO

pickle_off = open('../input/base04/X_val4s.pickle', 'rb')

X_val = pickle.load(pickle_off)



pickle_off = open('../input/base04/y_val4s.pickle', 'rb')

y_val = pickle.load(pickle_off)



# TESTE

pickle_off = open('../input/bases-rn/novas-bases/novas-bases/X_test.pickle', 'rb')

X_test = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/y_test.pickle', 'rb')

y_test = pickle.load(pickle_off)
scaler = MinMaxScaler()

X_test = scaler.fit_transform(X_test)
def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):

    accuracy = accuracy_score(y, y_pred_class)

    recall = recall_score(y, y_pred_class)

    precision = precision_score(y, y_pred_class)

    f1 = f1_score(y, y_pred_class)

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
# inicialize the class with two parameters

# Hidden_layer_sizes --> we create three layers of 10 nodes each

# max_iter --> number of iterations, or the epochs

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),activation='tanh', solver='sgd', 

                    learning_rate='adaptive', learning_rate_init= 0.25, max_iter = 1000, 

                    early_stopping=True, n_iter_no_change=5)

# use to train the algorithm on the training data

mlp.fit(X_train, y_train)



# predição no conjunto de validação

mlp_pred_val = mlp.predict(X_val)



# predição no conjunto de teste 

mlp_pred_test = mlp.predict(X_test)
# impressão das métricas de avaliação



mlp_pred_test_scores = mlp.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, mlp_pred_test, mlp_pred_test_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('Matriz de confusão')

skplt.metrics.plot_confusion_matrix(y_test, mlp_pred_test, normalize=True)

plt.show()



print('Curva ROC')

fp_mlp, tp_mlp, _ = roc_curve(y_test, mlp_pred_test)

plt.plot(fp_mlp, tp_mlp, label='MLP')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')
print("Acurácia (treinamento): {0:.3f}".format(mlp.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(mlp.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(mlp.score(X_test, y_test)))
rf = RandomForestClassifier(n_estimators=105, max_depth=100, min_samples_leaf=4, min_samples_split=10)



# treinamento dos dados

rf.fit(X_train, y_train)



# predição no conjunto de validação

rf_pred_val = rf.predict(X_val)



# predição no conjunto de teste

rf_pred_test = rf.predict(X_test)
# impressão das métricas de avaliação



rf_pred_test_scores = rf.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, rf_pred_test, rf_pred_test_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('\nMatriz de confusão')

skplt.metrics.plot_confusion_matrix(y_test, rf_pred_test, normalize=True)

plt.show()



print('Curva ROC')

fp_rf, tp_rf, _ = roc_curve(y_test, rf_pred_test)

plt.plot(fp_rf, tp_rf, label='RF')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')
print("Acurácia (treinamento): {0:.3f}".format(rf.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(rf.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(rf.score(X_test, y_test)))
gb = GradientBoostingClassifier(n_estimators=200, learning_rate= 0.75, max_features=64, 

                                max_depth = 2, random_state = 0)



# treino dos dados 

gb.fit(X_train, y_train)



# predição no conjunto de validação

gb_pred_val = gb.predict(X_val)



# predição no conjunto de teste

gb_pred_test = gb.predict(X_test)
# impressão das métricas de avaliação



gb_pred_test_scores = gb.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, gb_pred_test, gb_pred_test_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('Matriz de confusão')

skplt.metrics.plot_confusion_matrix(y_test, gb_pred_test, normalize=True)

plt.show()



print('Curva ROC')

fp_gb, tp_gb, _ = roc_curve(y_test, gb_pred_test)

plt.plot(fp_gb, tp_gb, label='GB')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')
print("Acurácia (treinamento): {0:.3f}".format(gb.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(gb.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(gb.score(X_test, y_test)))
ens_clf = VotingClassifier([('mlp', mlp), ('gb', gb), ('rf', rf)],voting='soft')



# treinamento dos dados

ens_clf.fit(X_train, y_train)



# predições nos conjuntos de teste e validação

ens_pred_test = ens_clf.predict(X_test)

ens_pred_val = ens_clf.predict(X_val)
# impressão das métricas de avaliação



ens_pred_test_scores = ens_clf.predict_proba(X_test)

ens_pred_val_scores = ens_clf.predict_proba(X_val)



accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, ens_pred_test, ens_pred_test_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('Matriz de confusão')

skplt.metrics.plot_confusion_matrix(y_test, ens_pred_test, normalize=True)

plt.show()



print('Curva ROC')

fp_gb, tp_gb, _ = roc_curve(y_test, ens_pred_test)

plt.plot(fp_gb, tp_gb, label='Ensemble')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')
print('Curva ROC')

#mlp

fpr_mlp, tpr_mlp, thresh_mlp = metrics.roc_curve(y_test, mlp_pred_test)

auc_mlp = metrics.roc_auc_score(y_test, mlp_pred_test)

plt.plot(fpr_mlp,tpr_mlp,label="mlp")



#randm forest

fpr_rf, tpr_rf, thresh_rf = metrics.roc_curve(y_test, rf_pred_test)

auc_rf = metrics.roc_auc_score(y_test, rf_pred_test)

plt.plot(fpr_rf,tpr_rf,label="rf")



#gradient booster

fpr_gb, tpr_gb, thresh_gb = metrics.roc_curve(y_test, gb_pred_test)

auc_gb = metrics.roc_auc_score(y_test, gb_pred_test)

plt.plot(fpr_gb,tpr_gb,label="gb")



#ensemble

fpr_ens, tpr_ens, thresh_ens = metrics.roc_curve(y_test, ens_pred_test)

auc_ens = metrics.roc_auc_score(y_test, ens_pred_test)

plt.plot(fpr_ens,tpr_ens,label="ens")



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')