# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

pd.set_option('display.max_columns',21)
data = pd.read_csv('../input/churn-prediction/Churn.csv')
df=data.copy()
df.head()
#target qui est une variable categorielle
target=df['Churn']
target.value_counts()
#1869 se sont desabonnés
#on va commencer par retirer la colonne de l'id qui ne nous servira à rien
df.drop('customerID',axis=1,inplace=True)
df.head()
df.shape  #7043 clients et 20 colonnes
df.dtypes
#Notre colonne TotalCharges est considéré comme un object,ce qui n'est pas normal
#on va le convertir en numerique avec la methode to_numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.dtypes
df.isna().sum()
#sur 7043 lignes on n'a que 11 valeurs manquantes dans la colonne TotalCharges,c'est probablement 
#à cause de ça que le type etait 'object'.
var_cat=df.select_dtypes(include='object') #variable categorielle
var_num=df.drop(var_cat.columns,axis=1)    #variable numerique
#Visualisation de la target
target.value_counts().plot.pie()
target.value_counts(normalize=True)
#26.5% des clients ont quittés l'entreprise
#Significations des variables
df.columns
var_num['Churn']=df['Churn']
var_num.head()
restés=var_num[var_num['Churn']=='No']
restés
restés_tenure=restés['tenure']
plt.hist(restés_tenure)
#On remarque que les plus anciens sont les plus nombreux à rester
#et les nouveaux viennent en seconde position
restés_tenure.mean() #la moyenne d'ancienneté de ceux qui sont restés 37.57
df['tenure'].mean() #la moyenne de l'ensemble 32.37
restés_tenure.mean()
#moyenne d'ancienneté resté>la moyenne d'ancienneté de l'ensemble
restés_SeniorCitizen=restés['SeniorCitizen']
plt.hist(restés_SeniorCitizen)
restés_SeniorCitizen.value_counts(normalize=True)
#parmis ceux qui sont restés,13% sont des SeniorCitizens=personnes agées
restés_MonthlyCharges=restés['MonthlyCharges']
plt.hist(restés_MonthlyCharges)
#ceux qui payent aux alentour de 20$ sont très nombreux
print(restés_MonthlyCharges.mean())
print(df['MonthlyCharges'].mean())
#l'ensemble a une moyenne mensuelle plus élévée que ceux qui sont restés
restés_TotalCharges=restés['TotalCharges']
plt.hist(restés_TotalCharges,bins=20)
#ceux qui ont un totalcharges inferieur à 1000 sont très nombreux
print(restés_TotalCharges.mean())
print(df['TotalCharges'].mean())
#ceux qui sont restés ont un totalcharges plus élévés que l'ensemble
partis=var_num[var_num['Churn']=='Yes']
partis
partis_tenure=partis['tenure']
plt.hist(partis_tenure)
print(df['tenure'].mean())
print(partis_tenure.mean())
print(restés_tenure.mean())
#Ceux qui sont partis ont une moyenne d'anciennetée< à la moitiée de la moyenne de tenure de ceux restés
partis_SeniorCitizen=partis['SeniorCitizen']
plt.hist(partis_SeniorCitizen)
partis_SeniorCitizen.value_counts(normalize=True)
#par contre là on remarque que les personnes agées representent 25 d'eux ceux qui sont partis
#contre 13% pour ceux qui sont restés
partis_MonthlyCharges=partis['MonthlyCharges']
plt.hist(partis_MonthlyCharges)
#ceux qui payent aux alentour de 20 ne sont plus les plus nombreux comparativement à restés_monthlycharges
print(restés_MonthlyCharges.mean())
print(df['MonthlyCharges'].mean())
print(partis_MonthlyCharges.mean())
#Ceux qui sont partis ont une mensualité moyenne plus élévée que les autres
partis_TotalCharges=partis['TotalCharges']
plt.hist(partis_TotalCharges,bins=20)
#ceux qui ont encore un total de moins de 1000$ sont très nombreux
print(restés_TotalCharges.mean())
print(df['TotalCharges'].mean())
print(partis_TotalCharges.mean())
#ceux qui sont partis la moyenne Totalcharges beaucoup moins élévée que pour les autres
var_cat.head()
restés_var_cat=var_cat[var_cat['Churn']=='No']
restés_var_cat
restés_var_cat.drop('Churn',axis=1,inplace=True)
restés_var_cat
for Col in restés_var_cat.columns:
    restés_var_cat[Col].value_counts().plot.pie()
    plt.show()
#les hommes sont majoritaires            #ceux qui ont un partenaires sont majoritaires

#Ceux qui n'ont pas de personnes à charges sont majoritaires     
#près de la moitiée n'est pas multiligne
#Ceux qui ont la fibre optique sont les plus nombreux
#la methode de paiement est presqu'equitablement retablit
partis_var_cat=var_cat[var_cat['Churn']=='Yes']
partis_var_cat
for Col in partis_var_cat.columns:
    partis_var_cat[Col].value_counts().plot.pie()
    plt.show()
for col in var_cat.columns:
    plt.figure()
    sns.heatmap(pd.crosstab(target,var_cat[col]),annot=True,fmt='d')
    #annot: pour pouvoir ecrire sur les figures
    #fmt='d': pour ecrire les valeurs en entier
df
sns.catplot(x='Partner',y='tenure',kind='boxen',data=df)
#Ceux qui ont des partenaires durent beaucoup plus
sns.catplot(x='Partner',y='TotalCharges',kind='boxen',data=df)
#comme ceux qui ont des partenaires durent plus,logiquement ils ont un montant total plus élévé
sns.catplot(x='Dependents',y='tenure',kind='boxen',data=df)
#ceux qui ont des personnes à charges sont plus fidèles
sns.catplot(x='Dependents',y='MonthlyCharges',kind='boxen',data=df)
sns.catplot(x='Dependents',y='TotalCharges',kind='boxen',data=df)
sns.catplot(x='MultipleLines',y='tenure',kind='boxen',data=df)
#parmis les nouveaux ceux qui n'ont pas de multilignes sont plus nombreux
#parmis les plus de 60 mois ,ceux qui ont multilignes sont plus nombreux
sns.catplot(x='MultipleLines',y='MonthlyCharges',kind='boxen',data=df)
#à partir de 80$ on a pas de client ayant le service telephone
#pour les clients qui payent plus de 80$ par mois les multilignes sont les plus nombreux
sns.catplot(x='MultipleLines',y='TotalCharges',kind='boxen',data=df)
#ceux qui ont multiligne sont  les plus nombreux parmis ceux qui payent plus de 4000$
#forte correlation entre l'ancienneté et Total charge
#ceux qui est logique,car plus le temps passe plus le cumul de ce qu'on paye augmente
sns.catplot(x='InternetService',y='MonthlyCharges',kind='boxen',data=df)
#ceux qui ont la fibre optique paye plus chère que les autres,puis ceux qui ont DSL
sns.catplot(x='InternetService',y='TotalCharges',kind='boxen',data=df)
sns.catplot(x='InternetService',y='tenure',kind='boxen',data=df)
sns.catplot(x='PaymentMethod',y='tenure',kind='boxen',data=df)
sns.catplot(x='StreamingTV',y='tenure',kind='boxen',data=df)
sns.catplot(x='Contract',y='tenure',kind='boxen',data=df)
df[df.TotalCharges.isnull()]
#on remarque que ses valeurs manquantes correspondent à de tout nouveaux clients qui n'ont pas encore un total 
#charges,donc soit on les remplace par 0 ou on retire ses clients de notre tableau
df.dropna(how='any',inplace=True)#pour supprimer les clients qui non pas encore une ancienneté d'un mois
df.isna().sum()
df.shape
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#features et target
features=df.drop('Churn',axis=1)
target=df['Churn']
cols=features.select_dtypes(include='object').columns
column_trans=make_column_transformer((OneHotEncoder(),cols),remainder='passthrough')
X=column_trans.fit_transform(features)
y=LabelEncoder().fit_transform(target)
y=y.ravel()
from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
Model=LogisticRegression()
Model.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix,classification_report

## Sur le train set
y_pred=Model.predict(X_train)
confusion_matrix(y_train,y_pred)
#3719:vrai positive,client  qui sont restés, et 406: faux positive(restés mais classés comme partis)
#814: vrai negative,client sont partis bien classés et 686: faux negatif(partis mais classé comme restés)
print(classification_report(y_train,y_pred,target_names=['restés','partis']))
### sur le test set
Y_pred=Model.predict(X_test)
confusion_matrix(y_test,Y_pred)
#936:vrai positive,client  qui sont restés, et 102: faux positive(restés mais classés comme partis)
#194: vrai negative,client sont partis bien classés et 175: faux negatif(partis mais classé comme restés)
print(classification_report(y_test,Y_pred,target_names=['restés','partis']))
Model.predict_proba(X_test) #probabilité pour chaque prediction
from sklearn.feature_selection import SelectKBest,chi2

Features=features
for col in features.select_dtypes(include='object').columns:
    Features[col]=LabelEncoder().fit_transform(features[col])
selector=SelectKBest(chi2,k=10)
selector.fit_transform(Features,y)
selector.get_support()
Columns=Features.columns[selector.get_support()]
Columns
#Standardisation
X1=Features[Columns]
X1=StandardScaler().fit_transform(X1)
X1
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y,test_size=0.2,random_state=0)
print(X1_train.shape,X1_test.shape)
print(y1_train.shape,y1_test.shape)
#GridSearchCV pour optimiser certains hyperparamètres
LogisticRegression()
params_grid={
    'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
    'C':[1,10,100,200],
    'multi_class':['auto','ovr','multinomial']
}
Grid=GridSearchCV(LogisticRegression(),params_grid,cv=5)
Grid.fit(X1_train,y1_train)
## Sur le train set
y1_pred=Grid.predict(X1_train)
confusion_matrix(y1_train,y1_pred)
print(classification_report(y1_train,y1_pred,target_names=['restés','partis']))
### sur le test set
Y1_pred=Grid.predict(X1_test)
confusion_matrix(y1_test,Y1_pred)
print(classification_report(y1_test,Y1_pred,target_names=['restés','partis']))
#les deux models nous donnent à peu près le meme resultat
