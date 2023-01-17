import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)  

import cufflinks as cf  

cf.go_offline() 

df = pd.read_csv('../input/glass.csv')

df.head()
df.iplot(kind='box')
sns.pairplot(df)
sns.pairplot(hue='Type',data=df)
for i in df.columns:

    print(i)

    df[i].iplot(kind='bar')
Type = df['Type'].value_counts().index.tolist()

Type_val = list(pd.value_counts(df['Type']))

explode = [0.1]*len(Type)

plt.figure(figsize=(10,10))

plt.pie(Type_val,explode=explode,labels=Type,autopct='%.1f%%')

plt.title('Types of metal distribution')

plt.show()
cols = df.columns.tolist()

cols.remove('Type')
Avg = df[cols[1:]].mean()

Avg = Avg.sort_values()

plt.figure(figsize=(10,7))

plt.barh(np.arange(len(cols[1:])), Avg.values, align='center')

plt.yticks(np.arange(len(cols[1:])), Avg.index)

plt.ylabel('Categories')

plt.xlabel('Average Predence')

plt.title('Average Presence of Each Metal')
fig, ax = plt.subplots(figsize=(20, 20)) 

sns.heatmap(df.corr(), annot = True, ax = ax)
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import scale as s

from scipy.cluster.hierarchy import dendrogram, linkage
def fd(*args, **kwargs):

    max_d = kwargs.pop('max_d', None)

    if max_d and 'color_threshold' not in kwargs:

        kwargs['color_threshold'] = max_d

    annotate_above = kwargs.pop('annotate_above', 0)



    ddata = dendrogram(*args, **kwargs)



    if not kwargs.get('no_plot', False):

        plt.title('Hierarchical Clustering Dendrogram (truncated)')

        plt.xlabel('sample index or (cluster size)')

        plt.ylabel('distance')

        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):

            x = 0.5 * sum(i[1:3])

            y = d[1]

            if y > annotate_above:

                plt.plot(x, y, 'o', c=c)

                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),

                             textcoords='offset points',

                             va='top', ha='center')

        if max_d:

            plt.axhline(y=max_d, c='k')

    return ddata
Z = sch.linkage(df.drop(['Type'],axis=1),'ward')

den = sch.dendrogram(Z)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 

plt.title('Hierarchical Clustering')
Z = linkage(df.drop(['Type'],axis=1),method='ward')

fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=20,max_d=26)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 
Z = linkage(df.drop(['Type'],axis=1),method='ward')

fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=8,max_d=18)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 
from sklearn.model_selection import train_test_split as t

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import sklearn.metrics as mt
x = df.iloc[:,:-1]

y = df.iloc[:,-1]

train_x,test_x,train_y,test_y = t(x,y,test_size=0.2)
rfc = RandomForestClassifier(n_estimators=50,max_depth=3)

rfc.fit(train_x,train_y)

print(f'Score = {rfc.score(test_x,test_y)}')
df_cm  = pd.DataFrame(mt.confusion_matrix(test_y,rfc.predict(test_x)),Type,Type)

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
print(mt.classification_report(test_y,rfc.predict(test_x)))
knnc = KNeighborsClassifier(n_neighbors=3)

knnc.fit(train_x,train_y)

print(f'Score = {mt.accuracy_score(test_y,knnc.predict(test_x))}')
df_cm  = pd.DataFrame(mt.confusion_matrix(test_y,knnc.predict(test_x)),Type,Type)

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
print(mt.classification_report(test_y,knnc.predict(test_x)))
xgbc = XGBClassifier()

xgbc.fit(train_x,train_y)

print(f'Score = {xgbc.score(test_x,test_y)}')
df_cm  = pd.DataFrame(mt.confusion_matrix(test_y,xgbc.predict(test_x)),Type,Type)

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
print(mt.classification_report(test_y,xgbc.predict(test_x)))
svc = SVC(kernel='linear',random_state=21)

svc.fit(train_x, train_y)

print(f'Score = {mt.accuracy_score(test_y,svc.predict(test_x))}')
df_cm  = pd.DataFrame(mt.confusion_matrix(test_y,svc.predict(test_x)),Type,Type)

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True)
print(mt.classification_report(test_y,svc.predict(test_x)))