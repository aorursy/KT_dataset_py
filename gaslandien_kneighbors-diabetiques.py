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
#importer les mains bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore') #pour ignorer les messages d'avertissement
%matplotlib inline
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
#importation de notre dat
df=data.copy()#copie de notre data
df.head()#les 5 premières lignes
#target ou la variable dependante
target=df['Outcome']
target.value_counts(normalize=True)#normlize=True pour l'avoir en pourcentage/100
target.value_counts().plot.pie()#representation graphique
#lignes et colonnes
df.shape
df.dtypes #types de nos variables
df.isna().sum()
df.corr() #correlation entre les differentes colonnes de notre df
df.describe() #quelques infos statistiques des colonnes
plt.figure(figsize=(12,7))
sns.heatmap(df.corr()) #representation graphiqe de la matrice de corrélation
sns.pairplot(df)
#merci seaborn!
for col in df.columns:
    sns.catplot(x='Outcome',y=col,kind='boxen',data=df)
    plt.show()
#visualisation graphique variable/target
df['Pregnancies'].value_counts()
#on voit qu'il une femme qui est tombée enceinte 17 fois,une autre 15...

df.drop('Outcome',axis=1).corr()
#aucune correlation n'est forte

plt.scatter(df['Pregnancies'],df['Age'],c=target)
#variables enceinte/age
plt.scatter(df['Insulin'],df['Glucose'],c=target)
plt.scatter(df['Insulin'],df['SkinThickness'],c=target)
X=df.drop('Outcome',axis=1)
y=target.ravel() #ravel() pour applatir notre y
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X) #standardiser notre X


from sklearn.model_selection import train_test_split,learning_curve,GridSearchCV

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#train_test_split nous permet de diviser nos données en deux parties, données d'entrainement et données pour le test de la
#performance

print(X_train.shape,X_test.shape)#la forme du train set
print(y_train.shape,y_test.shape)#la forme du test set
KNeighborsClassifier()
params_grid={'n_neighbors':np.arange(100),
            'metric':['minkowski','euclidean','chebyshev','manhattan']} 
#2 hyperparamètes, le nombre de voisin et comment identifier ses voisin à traver le metric à utiliser

Grid=GridSearchCV(KNeighborsClassifier(),params_grid,cv=5)
Grid.fit(X_train,y_train) #entrainement
Grid.best_params_ #meilleurs hyperparamètres

N,train_score,val_score=learning_curve(Grid,X_train,y_train,cv=4,train_sizes=np.linspace(0.1,1,10))
plt.figure(figsize=(12,7))
plt.plot(N,train_score.mean(axis=1),label='train_score')
plt.plot(N,val_score.mean(axis=1),label='validation_score')
plt.legend()

#va nous permettre de visualiser la performance de notre model en fonction de la quantité de données qu'on
#lui fournit

Model=Grid.best_estimator_#les meilleurs hyperparamètres pour notre model

y_pred=Model.predict(X_train)
Y_pred=Model.predict(X_test)
from sklearn.metrics import confusion_matrix,jaccard_similarity_score
#trainset
print(confusion_matrix(y_train,y_pred))
print(jaccard_similarity_score(y_train,y_pred))
#testset
print(confusion_matrix(y_test,Y_pred))
print(jaccard_similarity_score(y_test,Y_pred))
from sklearn.feature_selection import SelectKBest,chi2
#chi2 pour le test d'independance des Xi et notre target
features=df.drop('Outcome',axis=1)
chi2(features,target)
selector=SelectKBest(chi2,k=5)
selector.fit_transform(features,target)
selector.get_support()#True pour dire que cette colonne est retenue,false pour le contraire
COLS=features.columns[selector.get_support()] #boolean indexing,pour voir les colonnes concernées
COLS
X1=df[COLS]
y1=target.ravel()
X1=StandardScaler().fit_transform(X1) 

X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
print(X1_train.shape,X1_test.shape)
print(y1_train.shape,y1_test.shape)
Grid1=GridSearchCV(KNeighborsClassifier(),params_grid,cv=5)
Grid1.fit(X1_train,y1_train)
Grid1.best_params_
Model1=Grid1.best_estimator_
y1_pred=Model1.predict(X1_train)
Y1_pred=Model1.predict(X1_test)
print(confusion_matrix(y1_train,y1_pred))
print(jaccard_similarity_score(y1_train,y1_pred))
print(confusion_matrix(y1_test,Y1_pred))
print(jaccard_similarity_score(y1_test,Y1_pred))
