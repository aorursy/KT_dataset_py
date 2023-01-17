

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn import preprocessing
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

db = pd.read_csv('/kaggle/input/young-people-survey/responses.csv')
db.head()
pd.set_option('display.max_rows', 500)
cols = pd.DataFrame(db.columns)
cols
df1 = db[db.columns[19:30]] #these are the 'Movies' related questions that I want to use
df2 = db[db.columns[63:72]] #these are the 'Phobias' related questions that I want to use
df3 = df1.merge(df2, how = 'inner', left_index = True, right_index = True) #let's put them together
df3.head()
df3.isnull().sum()
df3_copy = df3


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')

imputer.fit(df3)

X = imputer.transform(df3)

db = pd.DataFrame(X)
db.isnull().sum() 
db.columns = df3_copy.columns
db.head()
db = np.array(db)

from kmodes.kmodes import KModes

cost = []
for nb_clusters in list(range(1,10)): 
    kmode = KModes(n_clusters = nb_clusters, init = 'Huang', n_init = 1, verbose = 1)
    kmode.fit_predict(db)
    cost.append(kmode.cost_)
y = np.array([i for i in range(1, 10, 1)]) 
plt.plot(y, cost) 
km = KModes(n_clusters = 4, init = 'Huang', n_init = 1, verbose = 1)
fitClusters = km.fit_predict(db)
db = df3_copy.reset_index()

clusters_df = pd.DataFrame(fitClusters)
clusters_df.columns = ['clusters_pred']
db_w_clusters = pd.concat([db, clusters_df], axis = 1).reset_index()

db_w_clusters = db_w_clusters.drop(['level_0', 'index'], axis = 1)

db_w_clusters.head()
import seaborn as sns

plt.subplots(figsize = (15,5))

sns.countplot(x=db_w_clusters['clusters_pred'],order=db_w_clusters['clusters_pred'].value_counts().index,hue=db_w_clusters['Movies'])
plt.show() 
db_gr = db_w_clusters.groupby('clusters_pred').mean()
db_gr
for col in db_gr.columns:
    plt.subplots(figsize = (15,5))
    sns.countplot(x=db_w_clusters['clusters_pred'],order=db_w_clusters['clusters_pred'].value_counts().index,hue=db_w_clusters[col])
    plt.show() 