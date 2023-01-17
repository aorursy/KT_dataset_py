# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Importing Data& EDA
beer_recipe = pd.read_csv('../input/recipeData.csv', index_col='BeerID', encoding='latin1')
beer_recipe.head(3)
beer_recipe.info()
beer_recipe.describe()
beer_recipe.dtypes
beer_recipe.describe().plot()
beer_recipe.columns
#finding missing values
null_vals=beer_recipe.isnull().sum()
beer_recipe.isnull().sum()
len(beer_recipe)
msno.matrix(beer_recipe)
sns.set()
null_vals.sort_values(inplace=True)
null_vals.plot(kind='bar',stacked=True,figsize=(20,10))
sns.despine(left=True, bottom=True)
bc=beer_recipe.corr()
plt.figure(figsize=(12,8))
sns.heatmap(bc,linewidth=0.5)
quantitative =[f for f in beer_recipe.columns if beer_recipe[f].dtype != 'object']

qualitative = [f for f in beer_recipe.columns if beer_recipe[f].dtype == 'object']

print(len(quantitative))
print(len(qualitative))
f = pd.melt(beer_recipe, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')

plt.figure(figsize=(10,10))
sns.jointplot(x='Color',y='BoilTime',data=beer_recipe,kind='hex')
plt.figure(figsize=(8,6))
sns.jointplot(x='ABV',y='MashThickness',data=beer_recipe,kind='kde',joint_kws={'color':'green'})
print( list(beer_recipe.select_dtypes(include=object).columns))
print(list(beer_recipe.select_dtypes(exclude=object).columns))
X=beer_recipe[['OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity', 'Efficiency', 'MashThickness','PitchRate']]
y=beer_recipe['StyleID']
#Data preprocssing
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
X=imputer.fit_transform(X)
#Spliting DATA into Train & TEST
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
#Standardaising
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X_train)
X=sc_x.fit(X_test)
sc_y=StandardScaler()
y=sc_y.fit_transform([y_train])
#importing classifier modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#cross validation
from sklearn.model_selection import KFold,cross_val_score
k_fold=KFold(n_splits=10,shuffle=True,random_state=0)
clf=KNeighborsClassifier()
scoring='accuracy'
score=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=RandomForestClassifier()
scoring='accuracy'
score=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=DecisionTreeClassifier()
scoring='accuracy'
score=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=GaussianNB()
scoring='accuracy'
score=cross_val_score(clf,X_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
#IN ABOVE ALL THE PREDICTIVE MODELS WE GOT RandomForestClassifier IS BEST AMONG THOSE BUT IT WAS ALSO A POOR MODEL EVENTHOUGH,WE GO FOR IT.
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#metrics calculation
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
cl=classification_report(y_test,y_pred)
print(cm)
print(cl)
#So finally our RandomForestClassifier model got only 48% accuracy,eventhough it is low