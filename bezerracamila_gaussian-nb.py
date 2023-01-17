# import packages

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
# open train df

df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
# check if there are missing values

# verificando se existem dados nulos

import missingno as msno

msno.matrix(df)
# print df head

df.head()
# print df shape

df.shape
# print df info

df.info()
#describe the df variables

df.describe().T
df.target.value_counts()
sns.countplot(y = df.target)
## correlation between target variable and the other columns

df_corr = df.corr().round(2)['target']
df_corr.sort_values(ascending =True)
df.head()
# I'll drop the ID_code and target variables of the dataset to do the PCA with the other variables.

#df_pca = df.drop(['ID_code'], axis = 1)

X_train = df.drop(['ID_code', 'target'], axis = 1)

X_test = df.drop(['ID_code', 'target'], axis = 1)
# importando as bibliotecas

from sklearn.preprocessing import StandardScaler 



# instanciando a variável

sc = StandardScaler() 





# ajustando com os dados de treino

X_train = sc.fit_transform(X_train) 

X_test = sc.fit_transform(X_test)
# importando as bibliotecas

from sklearn.decomposition import PCA



# instanciando o modelo

pca = PCA(n_components = 2)



# ajustando com os modelos de treino

X_train_pca = pca.fit_transform(X_train)



# transformando os dados de teste

X_test_pca = pca.transform(X_test)
X_train_pca
# Plotando um gráfico de dispersão entre as duas variáveis criadas pelo PC

# para identificaçào de como o PCA distribuiu as categorias

df2 = df.copy(deep=True)

# criação de duas novas colunas com as 2 dimensões do PCA

df2['PCA1'] = X_train_pca[:, 0]

df2['PCA2'] = X_train_pca[:, 1]

# criação de duas novas colunas com as 2 dimensões do PCA

df2['PCA1_test'] = X_test_pca[:, 0]

df2['PCA2_test'] = X_test_pca[:, 1]
fig, axs = plt.subplots(1,2)

# plotando uma dispersão das novas colunas diferenciando as espécies

sns.lmplot('PCA1', 'PCA2', hue='target', data=df2, fit_reg=False);

# plotando uma dispersão das novas colunas diferenciando as espécies

sns.lmplot('PCA1_test', 'PCA2_test', hue='target', data=df2, fit_reg=False);
df.target
# importando as bibliotecas

from sklearn.linear_model import LogisticRegression



# instanciando o modelo 

classifier = LogisticRegression(random_state = 42)

# ajustando o modelo

classifier.fit(X_train_pca, df.target) 
# predição de valores com dados de teste com a Regressão Logística

y_pred_pca = classifier.predict(X_train_pca) 
# plotando a Matriz de Confusão entre os valores reais e preditos



# importando a biblioteca

from sklearn.metrics import confusion_matrix 

# plotando a matriz

cm = confusion_matrix(df.target, y_pred_pca) 

cm
sns.heatmap(cm,annot=True,cbar=False, xticklabels='auto' )


# importar a funcao

from sklearn import metrics



# plotando a curva ROC

y_pred_proba_pca = classifier.predict_proba(X_train_pca)[::,1]

fpr, tpr, _ = metrics.roc_curve(df.target,  y_pred_proba_pca)

auc = metrics.roc_auc_score(df.target, y_pred_proba_pca)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
# importando as bibliotecas

from sklearn.preprocessing import StandardScaler 



# instanciando a variável

sc = StandardScaler() 





# ajustando com os dados de treino

X= df.drop(['ID_code', 'target'], axis = 1)

X = sc.fit_transform(X) 

y= df.target
# import the function

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
print(len(y_train))

print(len(X_train))

print(len(y_test))

print(len(X_test))
# importar a funcao

from sklearn.linear_model import LogisticRegression



# isntanciar o modelo

clf = LogisticRegression(dual = False, max_iter = 5000)



# ajustar aos dados de treino

clf.fit(X_train, y_train)



# predições para os dados de teste

y_pred = clf.predict(X_test)
# importar a funcao

from sklearn import metrics



confusion_matrix(y_test, y_pred)
# chamando a função da matriz de confusão

metrics.confusion_matrix
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
# plotando a curva ROC

y_pred_proba = clf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
len(df)
# fazendo um undersampling da classe com output zero (em maior número)

df_sample=df.sample(n=10000)
# importando as bibliotecas

from sklearn.preprocessing import StandardScaler 



# instanciando a variável

sc = StandardScaler() 





# ajustando com os dados de treino

X_sp= df_sample.drop(['ID_code', 'target'], axis = 1)

X_sp = sc.fit_transform(X_sp) 

y_sp= df_sample.target



from sklearn.model_selection import train_test_split



Xsp_train, Xsp_test, ysp_train, ysp_test = train_test_split(X_sp, y_sp, test_size = 0.2, random_state = 42, stratify = y_sp)
print(len(Xsp_train))

print(len(Xsp_test))
# importando as bibliotecas dos modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB



# definindo uma lista com todos os modelos

classifiers = [

    KNeighborsClassifier(),

    GaussianNB(),

    LogisticRegression(dual=False,max_iter=5000),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# rotina para instanciar, predizer e medir os resultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(Xsp_train, ysp_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Results****')

    ysp_pred = clf.predict(Xsp_test)

    print("Accuracy:", metrics.accuracy_score(ysp_test, ysp_pred))

    print("Precision:", metrics.precision_score(ysp_test, ysp_pred))

    print("Recall:", metrics.recall_score(ysp_test, ysp_pred))

    

     # plotando a curva ROC

    y_pred_proba = clf.predict_proba(X_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label=name+", auc="+str(auc))

    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

    plt.legend(loc=4)
# definindo variáveis para cada uma das classes

df_0 = df[df.target == 0]

df_1 = df[df.target==1]
print(len(df_0))

print(len(df_1))
# undersampling

df_0=df_0.sample(len(df_1))
df_concat = pd.concat([df_0,df_1])

df_concat.target.value_counts()
# ajustando com os dados de treino

X= df_concat.drop(['ID_code', 'target'], axis = 1)

X = sc.fit_transform(X) 

y = df_concat.target



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
# ignorando os warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# importnado as bibliotecas com os modelos classificadores



# definindo uma lista com todos os classificadores

classifiers = [

    KNeighborsClassifier(),

    GaussianNB(),

    LogisticRegression(),

    #SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# definindo o tamanho da figura para o gráfico

plt.figure(figsize=(12,8))



# rotina para instanciar, predizer e medir os rasultados de todos os modelos

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
# importar a funcao

from sklearn.naive_bayes import GaussianNB



# isntanciar o modelo

classifier = GaussianNB()



# ajustar aos dados de treino

classifier.fit(X_train, y_train)



# predições para os dados de teste

y_pred = clf.predict(X_test)
metrics.confusion_matrix



print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
# plotando a curva ROC

y_pred_proba = classifier.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')

plt.legend(loc=4)
# Verificando assimetria:

from scipy import stats



# choose numeric features

numeric_feats = df_concat.dtypes[df_concat.dtypes !="object"].index



skewed_feats = df_concat[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending = False)



print("\nAssimetria: \n")

skew_df = pd.DataFrame({'Skew' :skewed_feats})

skew_df.head(20)
norm = np.linalg.norm(df.var_179)

normal_array = df.var_179/norm

normal_array.skew()