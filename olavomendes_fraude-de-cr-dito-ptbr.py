import numpy as np 
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
import itertools

from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, ShuffleSplit, learning_curve, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(5)
print('Linhas: ', df.shape[0])
print('Colunas: ', df.shape[1])
df.describe().T
df.isnull().sum().max()
df.columns
print('Transações não fraudulentas: ', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% dos dados')
print('Transações fraudulentas: ', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% dos dados')
plt.figure(figsize=(8, 4))

sns.countplot('Class', data=df, palette=['b', 'r'])
plt.title('Distribuição da classe \n0: Não fraudulenta | 1: Fraudulenta')

plt.show()
# Reescala a distribuição para ficar com média 0 e desvio-padrão 1
std_scaler = StandardScaler()
# Utiliza o IQR (Inter Quartile Range) no escalonamento
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
# Armazena os dados das colunas criadas anteriormente
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

# Deleta as colunas criadas
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

# Insere os valores das colunas criadas na 1ª e 2ª colunas do dataframe, respectivamente
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

df.head()
# Features
X = df.drop('Class', axis=1)
# Target
y = df['Class']

# Separa os dados de maneira estratificada (mantendo as proporções originais)
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    
    # Armazena os dados originais
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
    # Transforma em array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values
    
    # Verifica se as distribuições de treino e teste são similares
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
    
    print('Distribuições:')
    print(train_counts_label/ len(original_ytrain))
    print(test_counts_label/ len(original_ytest))
    print('-' * 100)
# Pegar uma amostra aleatória. frac = 1 significa que a fração conterá todos os dados
df = df.sample(frac=1)

# Transações fraudulentas
fraud_df = df.loc[df['Class'] == 1]
# Transações não fraudulentas (a amostra terá tamanho de 492)
non_fraud_df = df.loc[df['Class'] == 0][:492]

# concatenando os dataframes anteriores
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# novo dataframe com a amostra aleatória dos dataframes anteriores. Novamente, a fração conterá todos os dados.
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
print('Distruição das classes no novo dataset')
print(new_df['Class'].value_counts()/len(new_df))

# Distruição das classes no dataframe balanceado
sns.countplot('Class', data=new_df, palette=['b', 'r'])
plt.title('Distruição das classes no novo dataset', fontsize=14)

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

# Correlação do dataframe original
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', vmin=-1, vmax=1,annot=True,  annot_kws={'size': 8}, ax=ax1)
ax1.set_title('Correlação dos dados desbalanceados', fontsize=14)

# Correlação do dataframe balanceado
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot=True, vmin=-1, vmax=1, annot_kws={'size': 8}, ax=ax2)
ax2.set_title('Correlação dos dados balanceados', fontsize=14)

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))
colors = ['b', 'r']

sns.boxplot(x="Class", y="V16", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Classe Negativa')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Classe Negativa')


sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Classe Negativa')


sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Classe Negativa')

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='r')
ax1.set_title('Distribuição V14 \n(transações fraudulentas)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='b')
ax2.set_title('Distribuição V12 \n(transações fraudulentas)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='g')
ax3.set_title('Distribuição V10 \n(transações fraudulentas)', fontsize=14)

plt.show()
# Remover os outliers de V14 (correlação negativa alta com a classe)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
# Valores do quartil 25 e quartil 75
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('QUARTIL 25: {} | QUARTIL 75: {}'.format(q25, q75))
# Interquartile range
v14_iqr = q75 - q25
print('IQR: ', v14_iqr)

# Limiar
v14_cut_off = v14_iqr * 1.5
# Limite superior e inferior
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('LIMIAR: ', v14_cut_off)
print('V14 LIMITE INFERIOR', v14_lower)
print('V14 LIMITE SUPERIOR', v14_upper)

# Ouliers (fora os limites estabelecidos anteriormente)
outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('V14 QUANTIDADE DE OUTLIERS EM FRAUDES:', len(outliers))

# Novo dataframe sem os outliers
new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 44)


# Remover os outliers de V12
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 LIMITE INFERIOR: {}'.format(v12_lower))
print('V12 LIMITE SUPERIOR: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 OUTLIERS: {}'.format(outliers))
print('V12 QUANTIDADE DE OUTLIERS EM FRAUDES: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('NÚMERO DE INSTÂNCIAS APÓS A REMOÇÃO DOS OUTLIERS: {}'.format(len(new_df)))
print('----' * 44)


# Remover os outliers de V10

v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 LIMITE INFERIOR: {}'.format(v10_lower))
print('V10 SUPERIOR: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 OUTLIERS: {}'.format(outliers))
print('V10 QUANTIDAADE DE OUTLIERS EM FRAUDES: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)


print('---' * 42)
print('NÚMERO DE INSTÂNCIAS APÓS A REMOÇÃO DOS OUTLIERS: {}'.format(len(new_df)))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

colors = ['b', 'r']

# V14
sns.boxplot(x='Class', y='V14', data = new_df, ax=ax1, palette=colors)
ax1.set_title('V14 \nRedução dos outliers', fontsize=14)

# V12
sns.boxplot(x='Class', y='V12', data = new_df, ax=ax2, palette=colors)
ax2.set_title('V12 \nRedução dos outliers', fontsize=14)

# V10
sns.boxplot(x='Class', y='V10', data = new_df, ax=ax3, palette=colors)
ax3.set_title('V10 \nRedução dos outliers', fontsize=14)


plt.show()
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# T-SNE
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# PCA
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

# TruncatedSVD
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
# Cria a figura
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
f.suptitle('Clusters usando Redução de Dimensionalidade', fontsize=14)

# Cores
blue = mpatches.Patch(color='b', label='Sem fraude')
red = mpatches.Patch(color='r', label='Fraude')


# t-SNE
ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y==0), cmap='coolwarm', label='Sem fraude', linewidths=2)
ax1.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=(y==1), cmap='coolwarm', label='Fraude', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

# Desenha uma grade nas figuras
ax1.grid(True)
ax1.legend(handles=[blue, red])

# PCA
ax2.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=(y==0), cmap='coolwarm', label='Sem fraude', linewidths=2)
ax2.scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=(y==1), cmap='coolwarm', label='Fraude', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)
ax2.legend(handles=[blue, red])


# TruncatedSVD
ax3.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c=(y==0), cmap='coolwarm', label='Sem Fraude', linewidths=2)
ax3.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c=(y==1), cmap='coolwarm', label='Fraude', linewidths=2)
ax3.set_title('SVD', fontsize=14)

ax3.grid(True)
ax3.legend(handles=[blue, red])


plt.show()
X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Transformar em array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
classifiers = {
    'LogisticRegression': LogisticRegression(),
    'KNearest': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier()
}
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    
    print('Classificador: ', classifier.__class__.__name__, 'possui um score de', round(training_score.mean(), 2) * 100, '%')
# Logistic Regression
# Parâmetros
log_reg_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# GridSearch
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
# Treinamento
grid_log_reg.fit(X_train, y_train)
# Melhores parâmetros
log_reg = grid_log_reg.best_estimator_



# Kneighbors
knears_params = {
    'n_neighbors': list(range(2, 5, 1)),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_



# SVC
svc_params = {
    'C': [0.5, 0.7, 0.9, 1],
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
}

grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_



# DecisionTree
tree_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(2, 4, 1)),
    'min_samples_leaf': list(range(5, 7, 1))
}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_
# Scores das validações cruzadas dos 4 modelos

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score: ', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score: ', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score: ', round(tree_score.mean() * 100, 2).astype(str) + '%')
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print('Train: ', train_index, 'Test:', test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

# Transformar em array
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values

# Listas para armazenar os scores
undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementação da técnica "NearMiss", apenas para ver a distribuição
X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)
print('Distribuição NearMiss: ', Counter(y_nearmiss))

# Validação cruzada da maneira correta
for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    # Cria a figura com 4 subplots
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    
    # Define os limites do eixo y
    if ylim is not None:
        plt.ylim(*ylim)
        
        
        # Primeiro "estimator"
        
        # Calcula a curva de aprendizado
        train_sizes, train_scores, test_scores = learning_curve(estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        # Média do treinamento
        train_scores_mean = np.mean(train_scores, axis=1)
        # Desvio-padrão do treinamento
        train_scores_std = np.std(train_scores, axis=1)
        # Média do teste
        test_scores_mean = np.mean(test_scores, axis=1)
        # Desvio-padrão do teste
        test_scores_std = np.std(test_scores, axis=1)
        # Preenche os eixos
        ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="#ff9124")
        ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        # Insere nos eixos os scores de treino
        ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",label="Score de treino")
        # Insere nos eixos os scores da validação cruzada
        ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Score da validação cruzada")
        # Título
        ax1.set_title("Logistic Regression", fontsize=14)
        # Legenda do eixo X
        ax1.set_xlabel('Training size (m)')
        # Legenda do eixo y
        ax1.set_ylabel('Score')
        # Desenha uma grande nos eixos
        ax1.grid(True)
        # Insere a legenda na melhor localização (onde não sobreponha as linhas desenhadas nos eixos)
        ax1.legend(loc="best")
        
        # Segundo "estimator" 
        train_sizes, train_scores, test_scores = learning_curve(estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax2.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
        ax2.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Score de treino")
        ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
        ax2.set_title("Knears Neighbors", fontsize=14)
        ax2.set_xlabel('Training size (m)')
        ax2.set_ylabel('Score')
        ax2.grid(True)
        ax2.legend(loc="best")

        
        # Terceiro "estimator"
        train_sizes, train_scores, test_scores = learning_curve(estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax3.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="#ff9124")
        ax3.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Score de treino")
        ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
        ax3.set_title("Support Vector Classifier", fontsize=14)
        ax3.set_xlabel('Training size (m)')
        ax3.set_ylabel('Score')
        ax3.grid(True)
        ax3.legend(loc="best")

        
        # Quarto "estimator"
        train_sizes, train_scores, test_scores = learning_curve(estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax4.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="#ff9124")
        ax4.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124", label="Score de treino")
        ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff", label="Cross-validation score")
        ax4.set_title("Decision Tree Classifier", fontsize=14)
        ax4.set_xlabel('Training size (m)')
        ax4.set_ylabel('Score')
        ax4.grid(True)
        ax4.legend(loc="best")
        
        # Retorna a figura
        return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=-1);
# Logistic Regression
log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method='decision_function')

# Knears Neighbors
knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

# SVC
svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method='decision_function')

# Decision Tree Classifier
tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
# False Positive Rate e True Positive Rate de todos os 4 classificadores
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    # Cria a figura
    plt.figure(figsize=(12,6))
    # Título
    plt.title('ROC Curve \n4 Classificadores', fontsize=18)
    # ROC AUC scores
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    # Linha central
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    # Legenda do eixo X
    plt.xlabel('False Positive Rate', fontsize=16)
    # Legenda do eixo y
    plt.ylabel('True Positive Rate', fontsize=16)
    
    plt.legend()
   

graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()
def log_roc_curve(log_fpr, log_tpr):
    # Cria a figura
    plt.figure(figsize=(10, 5))
    # Título
    plt.title('LogisticRegression ROC Curve', fontsize=16)
    
    # FPR e TPR na cor azul
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    # Linha central na cor vermelha
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01, 1, 0, 1])
    
    
log_roc_curve(log_fpr, log_tpr)
plt.show()
precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

# Predições
y_pred = log_reg.predict(X_train)

# Overfitting Case
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))

print('---' * 45)

# Como deveria ser
print('Como deveria ser:\n')
print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))
print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))
print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
# Média de precision-recall
undersample_y_score = log_reg.decision_function(original_Xtest)
undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Average precision-recall score: {0:0.2f}'.format(undersample_average_precision))
fig = plt.figure(figsize=(10, 5))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='b', alpha=0.3, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='r')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Undersampling Precision-Recall curve: \nAverage Precision-Recall score = {0:0.2f}'.format(undersample_average_precision), fontsize=16)
print('Tamanho do X (treino): {} | Tamanho do y (treino): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Tamanho do X (teste): {} | Tamanho do y (teste): {}'.format(len(original_Xtest), len(original_ytest)))

# Lista para armazenar os scores
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Parâmetros da Logistic Regression
log_reg_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Randomized SearchCV
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# Implementação do SMOTE
# Cross-validation da maneira correta
for train, test in sss.split(original_Xtrain, original_ytrain):
    # Pipeline
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE durante a validação cruzada
    # Treinamento do modelo
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    # Melhores parâmetros
    best_est = rand_log_reg.best_estimator_
    # Predições
    prediction = best_est.predict(original_Xtrain[test])
    
    # Armazena os "scores" nas listas criadas anteriormente
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
# Exibe os "scores"
print('---' * 40)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)
labels = ['Não fraudulenta', 'fraudulenta']

# Predição com SMOTE
smote_prediction = best_est.predict(original_Xtest)
# Printa a "classification report"
print(classification_report(original_ytest, smote_prediction, target_names=labels))
# Logistic Regression treinado com SMOTE
y_pred_log_reg = best_est.predict(X_test)

# Outros modelos com undersampling
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)

# Matriz de confusão de todos os modelos
log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

# Cria a figura e os axes
fig, ax = plt.subplots(2, 2,figsize=(12,6))

# Exibe a matriz de confusão do modelo Logistic Regression
sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.Blues)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

# Exibe a matriz de confusão do modelo Knears Neighbors
sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.Blues)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

# Exibe a matriz de confusão do modelo SVC
sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.Blues)
ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

# Exibe a matriz de confusão do modelo Decision Tree
sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.Blues)
ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.tight_layout()
plt.show()
# classification_reports de todos os modelos

print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('---' * 40)

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('---' * 40)

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('---' * 40)

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))
# Logistic Regression com "undersampling"
y_pred = log_reg.predict(X_test)
undersample_score = accuracy_score(y_test, y_pred)

# Logistic Regression com SMOTE
y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)

# Dicionário com os scores das duas técnicas (undersampling e oversampling)
d = {
    'Técnica': ['Random undersampling', 'Oversampling (SMOTE)'],
    'Score': [undersample_score, oversample_score]
}

# Cria um dataframe com o dicionário
final_df = pd.DataFrame(data=d)

# Armazena o "Score" em outra variável
score = final_df['Score']
# Remove a coluna "Score"
final_df.drop('Score', axis=1, inplace=True)
# Insere os dados armazenados anteriormente na segunda coluna
final_df.insert(1, 'Score', score)

final_df
# Tamanho da camada de entrada
n_inputs = X_train.shape[1]

# Criação da rede
undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
# Visualização da arquitetura da rede
undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True)
undersample_fraud_predictions = undersample_model.predict_classes(original_Xtest, batch_size=200)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Dados originais
undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(10,5))

fig.add_subplot(111)
plot_confusion_matrix(undersample_cm, labels, title="Random UnderSample \n Confusion Matrix", cmap=plt.cm.Reds)

# SMOTE
sm = SMOTE('minority', random_state=42)

# Treina os dados originais utilizando SMOTE
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)

# Modelo com os melhores parâmetros
log_reg_sm = grid_log_reg.best_estimator_
# Treina o modelo utilizando os dados
log_reg_sm.fit(Xsm_train, ysm_train)
n_inputs = Xsm_train.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)
# DAdos originais
oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)
labels = ['No Fraud', 'Fraud']

fig = plt.figure(figsize=(10,5))

fig.add_subplot(111)
plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)