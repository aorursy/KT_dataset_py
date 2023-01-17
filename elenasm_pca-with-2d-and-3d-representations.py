

import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



db = pd.read_csv('/kaggle/input/musicfeatures/data.csv')

db.head()
print(db.info())

print(db.shape)
db['label'].value_counts() #the genres are equally represented
db1 = db.iloc[:,1:29]

db1.info()
from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

db_pca = pca.fit_transform(db1)
pca.explained_variance_ratio_
principalDf = pd.DataFrame(data = db_pca

             , columns = ['principal component 1', 'principal component 2','principal component 3' ])
finalDf = pd.concat([principalDf, db['label']], axis = 1)

finalDf.head()
import plotly.express as px

fig = px.scatter(db_pca, x = 0, y = 1, color = finalDf['label'])

fig.show()
fig = px.scatter_3d(db_pca, x = 0, y = 1, z = 2, color = finalDf['label'])

fig.show()
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

features = db1.columns



fig = px.scatter(db_pca, x=0, y=1, color= finalDf['label'])

 

for i, feature in enumerate(features):

    fig.add_shape(

        type='line',

        x0=0, y0=0,

        x1=loadings[i, 0],

        y1=loadings[i, 1]

    )

    fig.add_annotation(

        x=loadings[i, 0],

        y=loadings[i, 1],

        ax=0, ay=0,

        xanchor="center",

        yanchor="bottom",

        text=feature,

    )

fig.show()