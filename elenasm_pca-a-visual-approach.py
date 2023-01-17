

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



db = pd.read_csv('/kaggle/input/glass/glass.csv')

db.head()
db.info()
corr_matrix = db.corr(method = 'pearson')

corr_matrix 
correlation = db.corr(method = 'pearson')

ax = sns.heatmap(correlation)
db1 = db.iloc[:,0:9]

db_target = db[["Type"]]

print(db1.head())

print(db_target.head())
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

db_pca = pca.fit_transform(db1)
pca.components_ 
df = pd.DataFrame(pca.components_, columns=list(db1.columns))

df.head()

pca.explained_variance_ratio_ 
main_db = pd.DataFrame(data = db_pca

             , columns = ['principal component 1', 'principal component 2'])
final_db = pd.concat([main_db, db_target], axis = 1)

final_db.head()


plt.figure(figsize=(8,6))

plt.scatter(db_pca[:,0],db_pca[:,1], c = final_db['Type'])





plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')



plt.legend(numpoints = 6)



plt.show()
#======== TEST WITH FUNCTION REPLICATING BIPLOT FUNCTION FROM R =========#



xvector = pca.components_[0] 

yvector = pca.components_[1]

xs = pca.transform(db1)[:,0] # see 'prcomp(my_data)$x' in R

ys = pca.transform(db1)[:,1]
for i in range(len(xvector)):



    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),

              color='r', width=0.0005, head_width=0.0025)

    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,

             list(db1.columns.values)[i], color='r')



for i in range(len(xs)):



    plt.plot(xs[i], ys[i], 'bo')

    plt.text(xs[i]*1.2, ys[i]*1.2, list(db1.index)[i], color='b')





plt.show()
#plt.close(fig) 

#fig.show('YourPathHere')