# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

# SeaBorn : librairie de graphiques avancés

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import plotly.graph_objs as go

import seaborn as sns

from plotly.offline import iplot
# Lecture des données d'apprentissage et de test

df1 = pd.read_csv("../input/student-alcohol-consumption/student-por.csv")

df2 = pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")
df1.head()
df1.count()
hommes = (df1.sex=="M")

femmes = (df1.sex=="F")
df1[hommes].head()  
fig = sns.FacetGrid(df1, hue="romantic", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "G3", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df1, hue="Walc", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "traveltime", shade=True)

fig.add_legend()
fig = sns.FacetGrid(df1, hue="Walc", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique

fig.map(sns.kdeplot, "absences", shade=True)

fig.add_legend()
df1[df1["school"] == 'GP']['reason'].value_counts().plot(kind='bar')
labels = df1["Fjob"].unique().tolist()

amount = df1["Fjob"].value_counts().tolist()



colors = ["orange", "green", "yellow", "white",'cyan']



trace = go.Pie(labels=labels, values=amount,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))



dt = [trace]

layout = go.Layout(title="Mother's job")



fig = go.Figure(data=dt, layout=layout)

iplot(fig, filename = 'pi_chart')
# creating a dict file  

gender = {'M': 1,'F': 2} 

yesno = {'yes': 1, 'no' : 0}

schooltoint = {'GP': 1, 'MS' : 2}

addresstoint = {'U' : 1, 'R' : 2}

famsizetoint = {'GT3' : 1, 'LE3' : 2}

Pstatustoint = {'T' : 1, 'A' : 2}

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

df1.sex = [gender[item] for item in df1.sex] 

df1.schoolsup = [yesno[item] for item in df1.schoolsup] 

df1.famsup = [yesno[item] for item in df1.famsup] 

df1.paid = [yesno[item] for item in df1.paid] 

df1.activities = [yesno[item] for item in df1.activities] 

df1.nursery = [yesno[item] for item in df1.nursery] 

df1.higher = [yesno[item] for item in df1.higher] 

df1.internet = [yesno[item] for item in df1.internet] 

df1.romantic = [yesno[item] for item in df1.romantic]

df1.school = [schooltoint[item] for item in df1.school]

df1.address = [addresstoint[item] for item in df1.address]

df1.famsize = [famsizetoint[item] for item in df1.famsize]

df1.Pstatus = [Pstatustoint[item] for item in df1.Pstatus]

df1 = df1.drop(['Mjob','Fjob','reason','guardian','Dalc'], axis=1)





df1.head() 
df2.sex = [gender[item] for item in df2.sex] 

df2.schoolsup = [yesno[item] for item in df2.schoolsup] 

df2.famsup = [yesno[item] for item in df2.famsup] 

df2.paid = [yesno[item] for item in df2.paid] 

df2.activities = [yesno[item] for item in df2.activities] 

df2.nursery = [yesno[item] for item in df2.nursery] 

df2.higher = [yesno[item] for item in df2.higher] 

df2.internet = [yesno[item] for item in df2.internet] 

df2.romantic = [yesno[item] for item in df2.romantic]

df2.school = [schooltoint[item] for item in df2.school]

df2.address = [addresstoint[item] for item in df2.address]

df2.famsize = [famsizetoint[item] for item in df2.famsize]

df2.Pstatus = [Pstatustoint[item] for item in df2.Pstatus]

df2 = df2.drop(['Mjob','Fjob','reason','guardian','Dalc'], axis=1)





df2.head() 
data_train = df1       # 80% des données avec frac=0.8

data_test = df2
X_train = data_train.drop(['Walc'], axis=1)

y_train = data_train['Walc']

X_test = data_test.drop(['Walc'], axis=1)

y_test = data_test['Walc']
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
from sklearn.metrics import accuracy_score

rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_

indices = np.argsort(importances)
plt.figure(figsize=(12,8))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), df1.columns[indices])

plt.title('Importance des caracteristiques')
df2 = df2.drop(['traveltime'], axis=1)

df1 = df1.drop(['traveltime'], axis=1)
data_train = df1      

data_test = df2
X_train = data_train.drop(['sex'], axis=1)

y_train = data_train['sex']

X_test = data_test.drop(['sex'], axis=1)

y_test = data_test['sex']
rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_

indices = np.argsort(importances)
plt.figure(figsize=(12,8))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), df1.columns[indices])

plt.title('Importance des caracteristiques')