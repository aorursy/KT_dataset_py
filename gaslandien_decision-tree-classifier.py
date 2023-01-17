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
#on importe les principales bibliothèques qui seront utilisées
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('../input/drug200/drug200.csv')
df=data.copy()
df.head()
target=df['Drug']
target
target.value_counts()
#le medicament Y a été prescrit à  91 malades,X à 54,malades,A à 23 malades,C à 16 malades et B à 16 malades
df.shape
#200 malades et 6 variables,5 independantes et la dependante qui est la target
df.dtypes
var_num=df[['Age','Na_to_K']]
var_cat=df[['Sex','BP','Cholesterol']]
df.isna().sum()
for col in var_cat.columns:
    print(var_cat[col].unique())
target.value_counts().plot.pie()
#Age: l'age du patient
#Sex: pour le sex du patient
#BP: blood pressure pour la pression sanguine
#Cholesterol: pour le taux de cholesterol
#Na_to_K:???
#Drug: notre target pour le medicament prescrit au final à un patient donné
var_num['target']=df['Drug']
var_num
sns.boxenplot(x='target',y='Age',data=var_num)
drugB=var_num[var_num['target']=='drugB']
drugB
#ceux qui ont prit le medicaments B ont tous plus de 50 ans
drugB['Age'].mean()
#avec une moyenne d'age de 62.5 ans
drugA=var_num[var_num['target']=='drugA']
drugA
drugA['Age'].mean()
#par contre ceux qui ont prit le medicament A sont jeunes avec une moyenne d'age de 35.9 ans
drugX=var_num[var_num['target']=='drugX']
drugY=var_num[var_num['target']=='drugY']
drugC=var_num[var_num['target']=='drugC']
print(drugX['Age'].mean())
print(drugY['Age'].mean())
print(drugC['Age'].mean())
#pour les medicaments X,Y,C ont a à peu près la même moyenne d'age
sns.boxenplot(x='target',y='Na_to_K',data=var_num)
#ceux qui ont prit le medicament Y ont un Na_to_K plus élévé
var_cat['target']=df['Drug']
var_cat
for col in var_cat.columns:
    plt.figure()
    sns.heatmap(pd.crosstab(target,var_cat[col]),annot=True,fmt='d')
df.columns
sns.heatmap(pd.crosstab(df['Cholesterol'],df['BP']),annot=True,fmt='d')
sns.heatmap(pd.crosstab(df['BP'],df['Sex']),annot=True,fmt='d')
sns.heatmap(pd.crosstab(df['Cholesterol'],df['Sex']),annot=True,fmt='d')
sns.boxenplot(x='BP',y='Na_to_K',data=df)
sns.boxenplot(x='Cholesterol',y='Na_to_K',data=df)
sns.boxenplot(x='Cholesterol',y='Age',data=df)
sns.boxenplot(x='BP',y='Na_to_K',data=df)
sns.boxenplot(x='Sex',y='Age',data=df)
df.describe()
df.describe(include='object')
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


features=df.drop('Drug',axis=1)
cols=features.select_dtypes(include='object').columns
column_trans=make_column_transformer((OneHotEncoder(),cols),remainder='passthrough')

#Encodage
X=column_trans.fit_transform(features)
y=LabelEncoder().fit_transform(target)
from sklearn.model_selection import train_test_split
y=y.ravel()#applatir notre y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
Model=DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#criterion
Model.fit(X_train,y_train)
#entrainement
#predictions
y_pred=Model.predict(X_train)
Y_pred=Model.predict(X_test)
from sklearn.metrics import accuracy_score
#metric pour le score
print(accuracy_score(y_train,y_pred))
print(accuracy_score(y_test,Y_pred))
#afficher les scores
