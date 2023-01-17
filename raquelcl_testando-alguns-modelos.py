import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = '../input/df-train/train.csv'
df_treino = pd.read_csv(train)

df_treino.head()
df_treino.info()
df_treino.describe()
sns.set_style('whitegrid') #gera um gride para receber as barras do gráfico

sns.countplot(x='AdoptionSpeed',data=df_treino,palette='RdBu_r') #definindo apaleta de cores azul e vermelho
sns.set_style('whitegrid')

sns.countplot(x='AdoptionSpeed',hue='Type',data=df_treino,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='AdoptionSpeed',hue='Vaccinated',data=df_treino,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='AdoptionSpeed',hue='Gender',data=df_treino,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='AdoptionSpeed',hue='MaturitySize',data=df_treino,palette='RdBu_r')
# Removendo variaveis para a elaboração do modelo:



df_treino.drop(['Name','RescuerID','Description','PetID','Dewormed','Fee'],axis=1,inplace=True)
df_treino.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_treino.drop('AdoptionSpeed',axis=1), 

                                                    df_treino['AdoptionSpeed'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
X_linear = df_treino[['Type', 'Age', 'Breed1','MaturitySize','Vaccinated','Sterilized','Health','Quantity','State','VideoAmt','PhotoAmt']]

y_linear = df_treino['AdoptionSpeed']
X_linear = df_treino[['Type', 'Age', 'Breed1','MaturitySize','Vaccinated','Sterilized','Health','Quantity','State','VideoAmt','PhotoAmt']]

y_linear = df_treino['AdoptionSpeed']
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train_linear,y_train_linear)
# Printando a intercepção

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X_train_linear.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test_linear)

plt.scatter(y_test_linear,predictions)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))