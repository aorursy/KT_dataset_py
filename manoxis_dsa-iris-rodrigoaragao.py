import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plot grafics
sns.set(color_codes=True)

import warnings # ignorar alertas desnecessários
warnings.filterwarnings('ignore')
df_treino = pd.read_csv('../input/iris-train.csv')
df_teste = pd.read_csv("../input/iris-test.csv")
df_treino.shape, df_teste.shape
df_treino.sample(10).T
df_teste.sample(10).T
df_treino.describe().T
df_treino.dtypes
df_treino.nunique()
df_treino.isnull().sum()
# verificando correlação entre as variáveis 

df_treino.corr()
# gerar um gráfico de dispersão simples
df_treino.plot(kind="scatter", x="SepalLengthCm", y='SepalWidthCm')
# gerar gráfico para analisar pares de características 
sns.pairplot(df_treino, hue='Species')
sns.heatmap(df_treino.corr(), annot=True, cmap='cubehelix_r')
df_treino.describe().T
sns.boxplot(x="Species", y='SepalLengthCm', data=df_treino)
#sns.boxplot(x="Species", y='SepalWidthCm', data=df_treino)
#sns.boxplot(x="Species", y='PetalLengthCm', data=df_treino)
#sns.boxplot(x="Species", y='PetalWidthCm', data=df_treino)

sns.violinplot(x="Species", y="SepalLengthCm", data=df_treino, hue='Species')
#sns.violinplot(x="Species", y="SepalWidthCm", data=df_treino, hue='Species')
#sns.violinplot(x="Species", y="PetalLengthCm", data=df_treino, hue='Species')
#sns.violinplot(x="Species", y="PetalWidthCm", data=df_treino, hue='Species')
feat = df_treino.iloc[:,0:-1]# Selecionando todas as linhas, da primeira coluna até a penúltima coluna.
targ = df_treino.iloc[:,-1] # Selecionando todas as linhas da última coluna ['Class'].


from sklearn.model_selection import train_test_split

feat_train, feat_test, targ_train, targ_test = train_test_split(feat, targ, random_state=42)


print('feat treino',feat_train.shape)
print('feat test',feat_test.shape)
print('targ treino',targ_train.shape)
print('targ test',targ_test.shape)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from xgboost import XGBClassifier - este caso tem que instalar antes via terminal


# medição da acurácia do modelo
from sklearn import metrics

logreg = LogisticRegression(random_state=42) # Criando o modelo

feat_train.sample(5)
logreg.fit(feat_train, targ_train) # Treinando o modelo
feat_test.sample(5)
pred = logreg.predict(feat_test) # predizendo

acc_logreg = round(metrics.accuracy_score(pred,targ_test)*100,1) # avaliando a acurácia. previsões x resultados reais
print("{}% de acurácia\n".format(acc_logreg,))
print(metrics.confusion_matrix(targ_test, pred))


#print("Gravando arquivo para submissão...\n")

#df_teste_lg = df_teste.copy()

#df_teste_lg['Species'] = logreg.predict(df_teste_lg)
#df_teste_lg[['Id','Species']].to_csv('logreg_rodrigoaragao.csv', index=False)
# GaussianNB

gaussian=GaussianNB() # Criando o modelo
gaussian.fit(feat_train, targ_train) # Treinando o modelo
pred=gaussian.predict(feat_test) # predizendo
acc_gaussian=round(metrics.accuracy_score(pred, targ_test)*100,1) # avaliando a acurácia. previsões x resultados reais
print(acc_gaussian,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))


#print("Gravando arquivo para submissão...\n")

#df_teste_gau = df_teste.copy()

#df_teste_gau['Species'] = logreg.predict(df_teste_gau)
#df_teste_gau[['Id','Species']].to_csv('gaussian_rodrigoaragao.csv', index=False)
# DecisionTreeClassifier

tree=DecisionTreeClassifier(random_state=42) # Criando o modelo
tree.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_tree=round(metrics.accuracy_score(pred,targ_test)*100,1) # avaliando a acurácia. previsões x resultados reais
print(acc_tree,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_tree = df_teste.copy()

df_teste_tree['Species'] = logreg.predict(df_teste_tree)
df_teste_tree[['Id','Species']].to_csv('tree_rodrigoaragao.csv', index=False)
# RandomForestClassifier

rfc=RandomForestClassifier(random_state=42) # Criando o modelo
rfc.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_rfc=round(metrics.accuracy_score(pred,targ_test)*100,1) # avaliando a acurácia. previsões x resultados reais
print(acc_rfc,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_rfc = df_teste.copy()

df_teste_rfc['Species'] = logreg.predict(df_teste_rfc)
df_teste_rfc[['Id','Species']].to_csv('rfc_rodrigoaragao.csv', index=False)
# ExtraTreesClassifier

etc=ExtraTreesClassifier(random_state=42) # Criando o modelo
etc.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_etc=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
print(acc_etc,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_etc = df_teste.copy()

df_teste_etc['Species'] = logreg.predict(df_teste_etc)
df_teste_etc[['Id','Species']].to_csv('etc_rodrigoaragao.csv', index=False)
# AdaBoostClassifier

abc=AdaBoostClassifier(random_state=42) # Criando o modelo
abc.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_abc=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
print(acc_abc,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_abc = df_teste.copy()

df_teste_abc['Species'] = logreg.predict(df_teste_abc)
df_teste_abc[['Id','Species']].to_csv('abc_rodrigoaragao.csv', index=False)
# SVC

svc=SVC(random_state=42) # Criando o modelo
svc.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_svc=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
print(acc_svc,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_svc = df_teste.copy()

df_teste_svc['Species'] = logreg.predict(df_teste_svc)
df_teste_svc[['Id','Species']].to_csv('svc_rodrigoaragao.csv', index=False)
# XGBClassifier

# TEM QUE INSTALAR...

#from xgboost import XGBClassifier

#xgb=XGBClassifier(random_state=42) # Criando o modelo
#xgb.fit(feat_train, targ_train) # Treinando o modelo
#pred=tree.predict(feat_test) # predizendo
#acc_xgb=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
#print(acc_xgb,"% de acurácia")
#print(metrics.confusion_matrix(targ_test, pred))
# GradientBoostingClassifier

gbc=GradientBoostingClassifier(random_state=42) # Criando o modelo
gbc.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_gbc=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
print(acc_gbc,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_gbc = df_teste.copy()

df_teste_gbc['Species'] = logreg.predict(df_teste_gbc)
df_teste_gbc[['Id','Species']].to_csv('gbc_rodrigoaragao.csv', index=False)
# KNeighborsClassifier

knn=KNeighborsClassifier() # Criando o modelo
knn.fit(feat_train, targ_train) # Treinando o modelo
pred=tree.predict(feat_test) # predizendo
acc_knn=round(metrics.accuracy_score(pred,targ_test)*100, 1) # avaliando a acurácia. previsões x resultados reais
print(acc_knn,"% de acurácia")
print(metrics.confusion_matrix(targ_test, pred))

df_teste_knn = df_teste.copy()

df_teste_knn['Species'] = logreg.predict(df_teste_knn)
df_teste_knn[['Id','Species']].to_csv('knn_rodrigoaragao.csv', index=False)
# importando cross_val_score
from sklearn.model_selection import cross_val_score
models = {'RandomForest': RandomForestClassifier(random_state=42),
          'ExtraTrees': ExtraTreesClassifier(random_state=42),
          'GBM': GradientBoostingClassifier(random_state=42),
          'DecisionTree': DecisionTreeClassifier(random_state=42),
          'AdaBoost': AdaBoostClassifier(random_state=42),
          'KNN 1': KNeighborsClassifier(n_neighbors=1),
          'KNN 3': KNeighborsClassifier(n_neighbors=3),
          'KNN 11': KNeighborsClassifier(n_neighbors=11),
          'SVC': SVC()}
          #'LinearRegression': LinearRegression()}
# feat_train, feat_test, targ_train, targ_test

def run_model(model, feat_train, feat_test, targ_train, targ_test):
    model.fit(feat_train, targ_train) # comando '.fit' é para treinar o modelo
    preds = model.predict(feat_test) # comeando '.predict' é para prever
    return round(metrics.accuracy_score(preds, targ_test)*100,1) # avaliando a acurácia. previsões x resultados reais

scores = []
for name, model in models.items():
    score = run_model(model, feat_train, feat_test, targ_train, targ_test)
    scores.append(score)
    print(name+' com a função:', score)
    
    score_cross = (cross_val_score(model, feat_train, targ_train, cv=10)) # 5 iterações
    names.append(name)
    scores.append(score_cross)
    

    print(name+' com cross_validation:', score_cross)
# plotando gráfico com os resultados

pd.Series(score, index=models.keys()).sort_values(ascending=False).plot.barh()
# Preparando a base para execução do modelo

feat_final = df_teste[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

# feat_final
# Executando a previsão na base de teste


df_final = df_teste.copy()

df_final['Species'] = logreg.predict(df_final)
df_final[['Id','Species']].to_csv('sub_rodrigoaragao.csv', index=False)
