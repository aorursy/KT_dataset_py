import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import csv 

import seaborn as sns

import plotly

import scipy.special

import xgboost as xgb 

import statsmodels.formula.api as smf

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.metrics import roc_curve, roc_auc_score

df_sample = pd.read_csv(r'/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
df_sample.head()
df_sample.tail()
df_sample.target.value_counts()
df_test_submission = pd.read_csv(r'/kaggle/input/santander-customer-transaction-prediction/test.csv')
df_test_submission.head()
list (df_test_submission.columns.values)
df_test_submission.shape
df_train0 = pd.read_csv(r'/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_train = df_train0.copy()
df_train.shape
#pd.set_option('display.max_columns', None) # option to display all columns

df_train.head()
# Looking at other Notebooks at Kaggle, I found this funtion from Gabriel Preda that shows the type of each feature and the percentage of null values in each of them.

#Source: https://www.kaggle.com/gpreda/santander-eda-and-prediction
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
%time

missing_data(df_train)
df_train.info()
#pd.set_option('display.max_columns', None)

df_train.describe()
df_train['target'].value_counts()
perc_0 = 100*(sum(df_train['target']==0))/df_train.shape[0]

perc_1 = 100*(sum(df_train['target']==1))/df_train.shape[0]

print ('Percentual de 0 = ',perc_0)

print ('Percentual de 1 = ', perc_1)
plt.figure(figsize=(3,4))

ax = sns.countplot(x = df_train.target, data = df_train)

df_train['target'].sample(1000,random_state=42).plot.kde()
#target_0 = df_train.loc[df_train['target']==0]

#target_1 = df_train.loc[df_train['target']==1]

#features = df_train.columns.values[2:202]



#for i in features:

    #plt.figure(figsize=(4,2))

    #plt.title("Distribution")

    #sns.distplot(target_0[i], kde=True, bins =60, color = 'blue', label = 'target_0')

    #sns.distplot(target_1[i], kde=True, bins =60, color = 'red', label = 'target_1')

    #sns.set(font_scale=0.8)

    #plt.legend()

    #plt.show()
#var = list (df_train.columns.values)

#var.remove('ID_code')

#var.remove('target')



#sample1 = df_train[df_train['target']==0].sample(500, random_state = 42)

#sample2 = df_train[df_train['target']==1].sample(500, random_state = 42)



# Plot

#for i in var:

    #plt.scatter(y=sample1[i], x=sample1[i].index, alpha=0.4)

    #plt.scatter(y=sample2[i], x=sample2[i].index, alpha=0.4, color = 'red')

    #plt.title(i)

    #plt.xlabel('indice')

    #plt.ylabel(i)

    #plt.show()
#for i in var:

    #f, ax = plt.subplots(figsize =(3,3))

    #ax = sns.boxplot(x=df_train['target'], y=df_train[i])
from sklearn.ensemble import RandomForestClassifier
X_train = df_train.drop(columns =[ 'ID_code','target'], axis=1)
y_train = df_train['target']
feat_labels = X_train.columns
feat_labels
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)



rf.fit(X_train, y_train)

importances = rf.feature_importances_



indices = np.argsort(rf.feature_importances_)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))
# importances: returns an array with the importance level of each feature, ordered by the column name 



# indices: return an array with the feature number, according to its importance, descending order 
indices
corrmat = df_train.corr()['target'].sort_values(ascending = False)

corrmat[0:40]                                                        
# 10 most positive correlated variables with the target
corrmat = df_train.corr()

cols = corrmat.nlargest(11,'target')['target'].index

cols
# 10 most negative correlated variables with the target
cols_negative = corrmat.nsmallest(10,'target')['target'].index

cols_negative
colunas_heat_map = []

for i in cols:

    colunas_heat_map.append (i)

for j in cols_negative:

    colunas_heat_map.append(j)



colunas_heat_map
corrmat = df_train[colunas_heat_map].corr()

sns.set(font_scale=1.15)

f, ax = plt.subplots(figsize=(20, 20))

hm = sns.heatmap(corrmat, 

                 cbar=True, # formatando a barra lateral de cores para o heatmap

                 annot=True, 

                 square=True, 

                 fmt='.2f', 

                 annot_kws={'size': 8}, 

                 yticklabels=corrmat.columns, 

                 xticklabels=corrmat.columns)
colunas = list(df_train.columns.values)

selected_columns=[]

contador = 1

while contador <= len(colunas):

    selected_columns =  colunas[(contador+1):(contador+12)]

    selected_columns.append('target')

    df1 = df_train[selected_columns].copy()

    sns.pairplot(data = df1.sample(100, random_state =42), height=1.8, hue = 'target')

    contador += 11

    
# Number of row of the original dataset df_train



qtd_linhas = df_train.shape[0]
qtd_linhas
# Function that excludes data that are bellow or above 3*standard deviations from the mean:

def exclui_extremos (nome_variavel):

    global df_train

    x_menos_3dv = df_train[nome_variavel].mean()-(3*df_train[nome_variavel].std())

    x_mais_3dv = df_train[nome_variavel].mean()+(3*df_train[nome_variavel].std())

    linhas_selecionadas = ((df_train[nome_variavel]<x_mais_3dv)&(df_train[nome_variavel]>x_menos_3dv))   

        

    df_train = df_train[linhas_selecionadas].copy()

    

    return 
colunas = list(df_train.columns.values)
colunas.remove('ID_code')

colunas.remove('target')
for i in colunas:

    exclui_extremos(i)
# New dataset without extreme values:

df_train.shape
# Percentage of data that has been preserved:



perc_preservado = df_train.shape[0]*100/qtd_linhas

perc_preservado
# Redefinition of X e y



X_train = df_train.drop(columns =[ 'ID_code','target'], axis=1)
y_train = df_train['target']
#First, let's separate the data in train and test.



# References that I used:

#https://beckernick.github.io/oversampling-modeling/

#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

#https://medium.com/analytics-vidhya/balance-your-data-using-smote-98e4d79fcddb

#https://towardsdatascience.com/a-deep-dive-into-imbalanced-data-over-sampling-f1167ed74b5
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
X_train.shape
X_train[0:5]
y_train.shape
y_train[0:5]
X_test.shape
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42, ratio = 1.0)

X_train_bal, y_train_bal = sm.fit_sample(X_train, y_train)



# ratio = 1, means that after the balancing we would like to have the same proportion of 1 and 0 for the target 

# default: k-neighbors = 5
# Verification of the balanced data
# X_train_bal:
X_train_bal.shape
type(X_train_bal)
X_train_bal[0:5]
# y_train_bal:
y_train_bal.shape
type(y_train_bal)
y_train_bal
# X_train_bal e y_train_bal possuem o mesmo número de linhas (ok)
# Verficando o número de target = 0 e target =1 
import collections, numpy

collections.Counter(y_train_bal)
# Ok, the data is balanced!
# I renamed the X_train and y_train with the balanced data
X_train = X_train_bal

y_train = y_train_bal
#importando a biblioteca

from sklearn.preprocessing import StandardScaler

#instanciando a variável

sc = StandardScaler()

#ajustando os dados de treino

X_train = sc.fit_transform(X_train)

#transformando os dados de teste

X_test = sc.transform(X_test)
# Agora temos:

# X_train com dados balanceados e na mesma escala

# X_test não balanceado e na mesma escala (aplicamos o scaling)

# y_test não balanceado (sem necessidade de scaling)

# y_train balanceado (sem necessidade de scaling)
# importando as bibliotecas

from sklearn.decomposition import PCA

for i in range (20,51,10):

    print ('Number of components = ', i)

    pca = PCA(n_components = i)

    X_train_pca = pca.fit_transform(X_train)

    print ('*'*30)

    print ('Noise Variance: ', pca.noise_variance_)

    print('Components: ', pca.components_)

    print ('*'*30)
pca.n_components
pca.explained_variance_ratio_
pca.explained_variance_
X_train[0]
# Verificando X_train_pca (principal components) após aplicação do PCA 
X_train_pca[0]
X_train_pca.shape
# transformando os dados de teste

X_test_pca = pca.transform(X_test)
X_test[0]  #primeira linha
# Verificando X_test_pca (principal components) após aplicação do PCA 
X_test_pca[0]
X_test_pca.shape
# Analisando quais são os "principal components" mais importantes
print(abs( pca.components_ ))
pca.components_.shape
# Primeiro principal component:
pca.components_[0]
# Encontrando qual o índice da feature mais importante em cada um dos 50 principal components:
most_important = [np.abs(pca.components_[i]).argmax() for i in range(50)]
most_important
# Identificando o nome das features mais importantes para cada principal component
# Nome de todas as colunas do df_train
feature_names = list(df_train.columns.values)
feature_names[0:5]
# exluindo o nome das features 'ID_code' e 'target' 
feature_names.remove('ID_code')
feature_names.remove('target')
len(feature_names)
# Imprimindo o nome da features mais importante para cada principal component
most_important_names = [feature_names[most_important[i]] for i in range(50)]
most_important_names
# References:

#https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis

# https://github.com/gaurav-kaushik/Data-Visualizations-Medium/blob/master/code/Interactive_PCA_and_Feature_Correlation.ipynb

# https://medium.com/cascade-bio-blog/creating-visualizations-to-better-understand-your-data-and-models-part-1-a51e7e5af9c0
# Variance from reconstructed data

reconstruct = pca.inverse_transform(X_train_pca)

differenceMatrix = X_train - reconstruct

differenceMatrix.var()



# Indicativo de quanta informação foi perdida com o PCA

# Quanto menor, melhor
reconstruct.shape
#definindo uma lista com os nomes das colunas

feature_cols = list(df_train.columns)

feature_cols.remove('target')

feature_cols.remove('ID_code')
# Imagem retirada do Notebook para reduzir o tamanho do arquivo.
## Feature importance

#plt.figure(figsize=(300,60))

#sns.heatmap(np.log(pca.inverse_transform(np.eye(X_train_pca.shape[1]))), cmap="hot", cbar=False, annot=True, xticklabels=df_train.columns)
# Agora vamos atribuir os valores de X_train e X_test com os valores do PCA
X_train= X_train_pca

X_test = X_test_pca
# Aplicando o pipeline de marchine learning 
# ignorando os warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# definindo uma lista com os nomes das colunas  ## feito anteriormente

#feature_cols = list(df_train.columns)

#feature_cols.remove('target')

#feature_cols.remove('ID_code')



# importnado as bibliotecas com os modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression   

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

# definindo uma lista com todos os classificadores

classifiers = [

    #KNeighborsClassifier(3),

    GaussianNB(),

    LogisticRegression(),

    #SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# definindo o tamanho da figura para o gráfico

plt.figure(figsize=(12,8))



# rotina para instanciar, predizer e medir os resultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(X_train, y_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Results****')

    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))

    

    

    # plotando a curva ROC

    y_pred_proba = clf.predict_proba(X_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label=name+", auc="+str(auc))

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

    plt.legend(loc=4)
# Melhores resultados foram com LogisticRegression e GradientBoostingClassifier, 

# porém o valor de precision ainda é bem baixo em ambos os modelos.
from sklearn import metrics
clf
# atribuindo os valores da matriz às variáveis 

# tp: True Positive

# fp: False Positive

# fn: False Negative

# tn: True Negative



tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()



tn, fp, fn, tp
# Cálculo da accuracy

# Representa  proporção de casos que foram corretamente previstos, 

#sejam eles verdadeiro positivo ou verdadeiro negativo.



accuracy = (tp+tn)/(tp+fp+tn+fn)

accuracy
# Cálculo do score de outra forma (accuracy)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
# Cálculo da precisão
precision = tp/(tp+fp)

precision
# Cálculo da Sensiblidade / Recall

# TPR: True Positive rate

# Representa a proporção de casos positivos que foram identificados corretamente.



recall = tp / (tp+fn)

recall
# Cálculo da specificity

# Representa proporção de casos negativos que foram identificados corretamente



specificity = tn/(tn+fp)

specificity
# Cálculo da prevalence



prevalence = (tp+fn)/(tp+fn+tn+fp)

prevalence
# Cálculo do F1 score

f1_gb = 2*(precision*recall)/(precision+recall)

f1_gb
#taxa de falsos positivos

false_positive = fp/(tn+fp)

false_positive
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

list(scores)
scores.mean()
mean_scores_GB = scores.mean()
# Temos que rodar novamente:
# Instanciando o modelo:

logreg = LogisticRegression().fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# atribuindo os valores da matriz às variáveis 

# tp: True Positive

# fp: False Positive

# fn: False Negative

# tn: True Negative



tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()



tn, fp, fn, tp
# Cálculo da accuracy

# Representa  proporção de casos que foram corretamente previstos, 

#sejam eles verdadeiro positivo ou verdadeiro negativo.



accuracy = (tp+tn)/(tp+fp+tn+fn)

accuracy
# Cálculo do score de outra forma (accuracy)
accuracy_score(y_test,y_pred)
# Cálculo da precisão
precision = tp/(tp+fp)

precision
# Cálculo da Sensiblidade / Recall

# TPR: True Positive rate

# Representa a proporção de casos positivos que foram identificados corretamente.



recall = tp / (tp+fn)

recall
# Cálculo da specificity

# Representa proporção de casos negativos que foram identificados corretamente



specificity = tn/(tn+fp)

specificity
# Cálculo da prevalence



prevalence = (tp+fn)/(tp+fn+tn+fp)

prevalence
# Cálculo do F1 score



f1 = 2*(precision*recall)/(precision+recall)

f1
#taxa de falsos positivos

false_positive = fp/(tn+fp)

false_positive
scores_logreg = cross_val_score(logreg, X_train, y_train, cv=5, scoring='accuracy')

list(scores_logreg)
mean_scores_logreg= scores_logreg.mean() 
mean_scores_logreg
# Data for submission
df_test_submission.head()
df_test_submission.shape
# Primeiro temos que definir X_test_submission
X_test_submission = df_test_submission.drop(columns =[ 'ID_code']).copy()
# Verificando a remoção da coluna
X_test_submission.head()
X_test_submission.shape
# Aplicando o scaling
#X_train1 = X_train_bal
#importando a biblioteca

#from sklearn.preprocessing import StandardScaler

#instanciando a variável

#sc = StandardScaler()

#ajustando os dados de treino

#X_train1 = sc.fit_transform(X_train1)

#transformando os dados de teste

X_test_submission = sc.transform(X_test_submission)
X_test_submission
X_test_submission.shape
# Aplicando o pca.transform em X_test_submission para redução da dimensionalidade
X_test_submission = pca.transform(X_test_submission)
X_test_submission
X_test_submission.shape
#Agora temos 50 features em vez de 200
# Calculando a probabilidade y predito=1 com a LogisticRegression
logreg 
y_pred_logreg = logreg.predict_proba(X_test_submission)
y_pred_logreg
y_pred_logreg = y_pred_logreg[:,:1]
y_pred_logreg
y_pred_logreg.shape
df_logReg= df_sample.copy()
df_logReg.head()
type(df_logReg)
df_logReg = df_logReg.drop(columns = ['target'])
df_logReg['target']=y_pred_logreg
df_logReg.head() 
df_logReg.to_csv('resultado_RegLog_04_10_pred_prob.csv', index=False)
teste = pd.read_csv('resultado_RegLog_04_10_pred_prob.csv')

teste.head()
# Após submissão do arquivo no Kaggle do modelo com Regressão Logística:
# Private Score: 0.85786

# Public Score: 0.85806