import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





train = pd.read_csv("../input/train.csv").drop('id', axis=1)



y_train = train['target']

X_train = train.drop('target', axis=1)



test = pd.read_csv('../input/test.csv')

X_test = test.drop('id', axis = 1)

# PCA in the training data


pca = PCA(n_components=2)

PC_train = pca.fit_transform(X_train)

principalDf = pd.DataFrame(data = PC_train

                           ,columns = ['pc1', 'pc2'])



final_pc_train_Df = pd.concat([principalDf, y_train], axis = 1)



# Create a trace

trace = go.Scatter(

    x = principalDf['pc1'],

    y = principalDf['pc2'],

    mode = 'markers',

    marker = dict(

        size=10,

        color = y_train,

    )

)

data = [trace]



layout = dict(title = 'PC on the training set',

              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)



init_notebook_mode(connected=True)

iplot(fig)

# PCA on traning+test data
train_test_id = pd.Series(np.concatenate((np.repeat(10, X_train.shape[0]), np.repeat(1, X_test.shape[0]))))

X_tot = pd.concat([X_train, X_test], axis = 0)



pca = PCA(n_components=2)

PC_tot = pca.fit_transform(X_tot)



principalDf = pd.DataFrame(data = PC_tot

                           ,columns = ['pc1', 'pc2'])



final_pc_tot_Df = pd.concat([principalDf, train_test_id], axis = 1)



# Create a trace

train_trace1 = go.Scatter(

    x = principalDf.iloc[0:X_train.shape[0],]['pc1'],

    y = principalDf.iloc[0:X_train.shape[0],]['pc2'],

    mode = 'markers',

    name='train',

    marker = dict(

        size=10,

        color = 'blue',

    )

)



test_trace2 = go.Scatter(

    x = principalDf.iloc[(X_train.shape[0]+1):principalDf.shape[0],]['pc1'],

    y = principalDf.iloc[(X_train.shape[0]+1):principalDf.shape[0],]['pc2'],

    mode = 'markers',

    name='test',

    marker = dict(

        size=1,

        color = ['red'],

    )

)





data = [train_trace1, test_trace2]



layout = dict(title = 'PC on the training set',

              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False),

              showlegend=True

             )



fig = dict(data = data, layout = layout)



init_notebook_mode(connected=True)

iplot(fig)



## add shape to see if the lables also match
# given the size of traning data is it difficult to specify the strenght of the covariance shift.