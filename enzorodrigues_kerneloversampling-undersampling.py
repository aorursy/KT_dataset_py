import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# Carregar o Dataset
cancer = load_breast_cancer()
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                  columns= np.append(cancer['feature_names'], ['target']))
# Informações dos dados
df.info()
df.head()
import seaborn as sns
ax = sns.countplot(x='target', data=df)
# Olhando a variável target
df['target'].value_counts()
X=df.drop('target',axis=1)
y=df.target
# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#train, test = train_test_split(df, test_size=0.3, random_state=42)
X_train.shape, X_test.shape
# Criar a lista de colunas para treino
#feats = [c for c in train.columns if c not in ['target']]
# definir um modelo e treina-lo
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
predi_test = xgb.predict(X_test)
#xgb.fit(train[feats], train['target'])
#predi_test = xgb.predict(test[feats])
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(accuracy_score(y_test,predi_test))
print(classification_report(y_test, predi_test))

#print(accuracy_score(test['target'], predi_test))
#print(classification_report(test['target'], predi_test))
print(confusion_matrix(y_test, predi_test))
import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
# Separar os dados
#X,y = df[feats], df[['target']]
# Usar tecnicas para balancear o dataset
# Começando com o Oversampling
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X_train,y_train)
xgb = XGBClassifier()
xgb.fit(X_ros, y_ros)
pred_test = xgb.predict(X_test)
# Juntar os dados
#df_over = pd.concat([X_ros, y_ros], axis=1)
#train, test = train_test_split(df_over, test_size=0.3, random_state=42)
#xgb.fit(train[feats], train['target'])
#pred_test = xgb.predict(test[feats])


#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#Instanciar o algoritmo
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))

#print(accuracy_score(test['target'], pred_test))
#print(classification_report(test['target'], pred_test))
print(confusion_matrix(y_test, pred_test))
# Utilizando o SMOTE
sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X_train,y_train)
xgb = XGBClassifier()
xgb.fit(X_sm, y_sm)
preds_test = xgb.predict(X_test)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#Instanciar o algoritmo


#df_over = pd.concat([X_sm, y_sm], axis=1)
#train, test = train_test_split(df_over, test_size=0.3, random_state=42)
# Treinar o modelo
#xgb.fit(train[feats], train['target'])
# Fazer previsão
#preds_test = xgb.predict(test[feats])
print(accuracy_score(y_test, preds_test))
print(classification_report(y_test, preds_test))

#print(accuracy_score(test['target'], preds_test))
#print(classification_report(test['target'], preds_test))
print(confusion_matrix(y_test, preds_test))
# Utilizando o TomekLinks
tl = TomekLinks()
X_tl,y_tl = tl.fit_resample(X_train,y_train)
xgb = XGBClassifier()
xgb.fit(X_tl, y_tl)
prds_test = xgb.predict(X_test)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#Instanciar o algoritmo


#df_under = pd.concat([X_tl, y_tl], axis=1)
#train, test = train_test_split(df_under, test_size=0.3, random_state=42)
# Treinar
#xgb.fit(train[feats], train['target'])
# Fazer previsão
#prds_test = xgb.predict(test[feats])
print(accuracy_score(y_test, prds_test))
print(classification_report(y_test, prds_test))

#print(accuracy_score(test['target'], prds_test))
#print(classification_report(test['target'], prds_test))
print(confusion_matrix(y_test, prds_test))
# Utilizando o RandomUnderSampler
rus = RandomUnderSampler()
# Caso tenha erro no "fit_resample" é pq esse método foi introduzido tardiamente para o imbalanced-learn API.
# Pode também usar o "fit_sample".
X_rus,y_rus = rus.fit_resample(X_train,y_train)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
#Instanciar o algoritmo
xgb = XGBClassifier()
xgb.fit(X_rus, y_rus)
pred_test = xgb.predict(X_test)
print(accuracy_score(y_test, pred_test))
print(classification_report(y_test, pred_test))
print(confusion_matrix(y_test, pred_test))
# Utilizando o NearMiss
nm = NearMiss()
X_nm, y_nm = nm.fit_resample(X_train, y_train)
xgb = XGBClassifier()
xgb.fit(X_nm, y_nm)
y_pred = xgb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predito'], margins=True))