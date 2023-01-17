import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import roc_curve

from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.neural_network import MLPClassifier



import pickle



import os



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt
# import os

# print(os.listdir('../input/bases-rn/novas-bases/novas-bases'))



# Carregando dados de teste 



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/X_test.pickle', 'rb')

X_test = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/y_test.pickle', 'rb')

y_test = pickle.load(pickle_off)



# Carregando dados de treino



pickle_off = open('../input/base03/X_train3s.pickle', 'rb')

X_train = pickle.load(pickle_off)



pickle_off = open('../input/base03/y_train3s.pickle', 'rb')

y_train = pickle.load(pickle_off)



# Carregando dados de validação



pickle_off = open('../input/base03/X_val3s.pickle', 'rb')

X_val = pickle.load(pickle_off)



pickle_off = open('../input/base03/y_val3s.pickle', 'rb')

y_val = pickle.load(pickle_off)
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
from sklearn.neural_network import MLPClassifier





mlp = MLPClassifier()



# use to train the algorithm on the training data

mlp.fit(X_train, y_train)



# make predictions on the data

mlp_pred_val = mlp.predict(X_val)

mlp_pred_test = mlp.predict(X_test)
mlp_pred_val_scores = mlp.predict_proba(X_val)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, mlp_pred_val, mlp_pred_val_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('Matriz de confusão')

skplt.metrics.plot_confusion_matrix(y_val, mlp_pred_val, normalize=True)

plt.show()

# conf_mat = confusion_matrix(y_val, mlp_pred_val)

# sns.heatmap(conf_mat, square=True, annot=True, cbar=False)

# plt.xlabel('predicted value')

# plt.ylabel('true value');
print("Acurácia (treinamento): {0:.3f}".format(mlp.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(mlp.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(mlp.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()



# treino dos dados

rf.fit(X_train, y_train)



# predição no conjunto de avaliação

rf_pred_val = rf.predict(X_val)



# predição no conjunto de teste

rf_pred_test = rf.predict(X_test)
rf_pred_val_scores = rf.predict_proba(X_val)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, rf_pred_val, rf_pred_val_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('\nMatriz de confusão')

skplt.metrics.plot_confusion_matrix(y_val, rf_pred_val, normalize=True)

plt.show()

# conf_mat = confusion_matrix(y_val, rf_pred_val)

# sns.heatmap(conf_mat, square=True, annot=True, cbar=False)

# plt.xlabel('predicted value')

# plt.ylabel('true value');
print("Acurácia (treinamento): {0:.3f}".format(rf.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(rf.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(rf.score(X_test, y_test)))
gb = GradientBoostingClassifier()



# treino dos dados

gb.fit(X_train, y_train)



# predição no conjunto de validação

gb_pred_val = gb.predict(X_val)



# predição no conjunto de teste

gb_pred_test = gb.predict(X_test)
# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

# for learning_rate in learning_rates:

#     gb = GradientBoostingClassifier(n_estimators=20, learning_rate= learning_rate, max_features=2, max_depth = 2, random_state = 0)

#     gb.fit(X_train, y_train)

#     gb_pred_val = gb.predict(X_val)

#     gb_pred_test = gb.predict(X_test)

#     print("###  Learning rate: ", learning_rate, '  ###')

#     print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))

#     print("Accuracy score (validation): {0:.3f}".format(gb.score(X_val, y_val)))

#     print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, y_test)))
# estimators = [10, 20, 50, 100, 150, 200]

# for estimator in estimators:

#     gb = GradientBoostingClassifier(n_estimators=estimator, learning_rate= 1, max_features=2, max_depth = 2, random_state = 0)

#     gb.fit(X_train, y_train)

#     gb_pred_val = gb.predict(X_val)

#     gb_pred_test = gb.predict(X_test)

#     print("###  estimator: ", estimator, '  ###')

#     print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))

#     print("Accuracy score (validation): {0:.3f}".format(gb.score(X_val, y_val)))

#     print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, y_test)))
# features = [2,4,8,16,32,64]

# for feature in features:

#     gb = GradientBoostingClassifier(n_estimators=10, learning_rate= 1, max_features=feature, max_depth = 2, random_state = 0)

#     gb.fit(X_train, y_train)

#     gb_pred_val = gb.predict(X_val)

#     gb_pred_test = gb.predict(X_test)

#     print("###  feature: ", feature, '  ###')

#     print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))

#     print("Accuracy score (validation): {0:.3f}".format(gb.score(X_val, gb_pred_val)))

#     print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, gb_pred_test)))
# gb = GradientBoostingClassifier(n_estimators=200, learning_rate= 0.75, max_features=64, max_depth = 2, random_state = 0)

# gb.fit(X_train, y_train)

# gb_pred_val = gb.predict(X_val)

# gb_pred_test = gb.predict(X_test)
gb_pred_val_scores = gb.predict_proba(X_val)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, gb_pred_val, gb_pred_val_scores)



print('Performance no conjunto de teste:')

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)



print('Matriz de confusão')

skplt.metrics.plot_confusion_matrix(y_val, gb_pred_val, normalize=True)

plt.show()

# conf_mat = confusion_matrix(y_val, gb_pred_val)

# sns.heatmap(conf_mat, square=True, annot=True, cbar=False)

# plt.xlabel('predicted value')

# plt.ylabel('true value');
print("Acurácia (treinamento): {0:.3f}".format(gb.score(X_train, y_train)))

print("Acurácia (validação): {0:.3f}".format(gb.score(X_val, y_val)))

print("Accuracy (teste): {0:.3f}".format(gb.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier

# mlp_ens = MLPClassifier(hidden_layer_sizes=(242,30,30,30),activation='tanh', max_iter = 1000, early_stopping=True)

# rf_ens = RandomForestClassifier()

# gb_ens = GradientBoostingClassifier(n_estimators=10, learning_rate= 1, max_features=16, max_depth = 2, random_state = 0)



ens_clf = VotingClassifier([('mlp', mlp), ('gb', gb), ('rf', rf)],voting='soft')

ens_clf.fit(X_train, y_train)

ens_pred_class = ens_clf.predict(X_val)

ens_pred_scores = ens_clf.predict_proba(X_val)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, ens_pred_class, ens_pred_scores)

print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#métricas de avaliação

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