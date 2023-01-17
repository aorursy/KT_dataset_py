!pip install eif

!pip install pyod
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import eif as iso

from sklearn.ensemble import IsolationForest

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline
mean = [0, 0, 0]

cov = [[4, 2, 0],

      [0, 1, 0],

      [0, 0, 1]]



x, y, z = np.random.multivariate_normal(mean, cov, 2500).T



x = x.reshape(2500, 1)

y = y.reshape(2500, 1)

z = z.reshape(2500, 1)

dataframe = pd.DataFrame(np.concatenate((x, y, z), axis = 1), columns = ['f1', 'f2', 'f3'])

X = dataframe.values
fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(dataframe.f1, dataframe.f2, dataframe.f3, c='b', marker='o')



ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()
# import graph objects as "go"

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.f1,

    y=dataframe.f2,

    z=dataframe.f3,

    mode='markers',

    marker=dict(

        size=2,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
anomalies_ratio = 0.03



if_sk = IsolationForest(n_estimators = 500, 

                        max_samples = 256,

                        contamination = anomalies_ratio, 

                        behaviour= " new", 

                        random_state = np.random.RandomState(42))

if_sk.fit(X)

y_pred = if_sk.predict(X)

y_pred = [1 if x == -1 else 0 for x in y_pred]
dataframe['target'] = y_pred

dataframe.head()
dataframe.target.value_counts()
predicted_normal = dataframe[dataframe['target']==0]

predicted_anomaly = dataframe[dataframe['target']==1]
fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(predicted_normal.f1, predicted_normal.f2, predicted_normal.f3, c='b', marker='o')

ax.scatter(predicted_anomaly.f1, predicted_anomaly.f2, predicted_anomaly.f3, c='r', marker='o')





ax.set_xlabel('X Label')

ax.set_ylabel('Y Label')

ax.set_zlabel('Z Label')



plt.show()
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=predicted_normal.f1,

    y=predicted_normal.f2,

    z=predicted_normal.f3,

    mode='markers',

    marker=dict(

        size=5,

        color='red',                # set color to an array/list of desired values      

    )

)



trace2 = go.Scatter3d(

    x=predicted_anomaly.f1,

    y=predicted_anomaly.f2,

    z=predicted_anomaly.f3,

    mode='markers',

    marker=dict(

    size=5,

    color='blue',

    )

)



data = [trace1, trace2]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
if_eif = iso.iForest(X, ntrees = 100, sample_size = 256, ExtensionLevel = 1)

anomaly_scores = if_eif.compute_paths(X_in = X)

# sort the scores

anomaly_scores_sorted = np.argsort(anomaly_scores)

# retrieve indices of anomalous observations

indices_with_preds = anomaly_scores_sorted[-int(np.ceil(anomalies_ratio * X.shape[0])):]

# create predictions 

y_pred = np.zeros_like(y)

y_pred[indices_with_preds] = 1
dataframe['target'] = y_pred

predicted_normal = dataframe[dataframe['target']==0]

predicted_anomaly = dataframe[dataframe['target']==1]
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=predicted_normal.f1,

    y=predicted_normal.f2,

    z=predicted_normal.f3,

    mode='markers',

    marker=dict(

        size=5,

        color='red',                # set color to an array/list of desired values      

    )

)



trace2 = go.Scatter3d(

    x=predicted_anomaly.f1,

    y=predicted_anomaly.f2,

    z=predicted_anomaly.f3,

    mode='markers',

    marker=dict(

    size=5,

    color='blue',

    )

)



data = [trace1, trace2]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
from pyod.utils.data import generate_data, get_outliers_inliers
X, y = generate_data(n_train=500, train_only=True, contamination=0.03, n_features=3)
dataframe = pd.DataFrame(np.concatenate((X, y.reshape(500,1)), axis=1), columns = 'x y z target'.split())
dataframe['target'] = dataframe['target'].astype(int)

inliers = dataframe[dataframe['target']==0]

outliers = dataframe[dataframe['target']==1]
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=inliers.x,

    y=inliers.y,

    z=inliers.z,

    mode='markers',

    marker=dict(

        size=5,

        color='blue',                # set color to an array/list of desired values      

    )

)



trace2 = go.Scatter3d(

    x=outliers.x,

    y=outliers.y,

    z=outliers.z,

    mode='markers',

    marker=dict(

    size=5,

    color='red',

    )

)



data = [trace1, trace2]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
if_eif = iso.iForest(X, ntrees = 100, sample_size = 256, ExtensionLevel = 1)

anomaly_scores = if_eif.compute_paths(X_in = X)

# sort the scores

anomaly_scores_sorted = np.argsort(anomaly_scores)

# retrieve indices of anomalous observations

indices_with_preds = anomaly_scores_sorted[-int(np.ceil(anomalies_ratio * X.shape[0])):]

# create predictions 

y_pred = np.zeros_like(y)

y_pred[indices_with_preds] = 1
dataframe['ai_pred'] = y_pred.astype(int)
dataframe.head()
true_positives = dataframe[(dataframe['target']==1) & (dataframe['ai_pred']==1)]

true_negatives = dataframe[(dataframe['target']==0) & (dataframe['ai_pred']==0)]

false_positives = dataframe[(dataframe['target']==0) & (dataframe['ai_pred']==1)]

false_negatives = dataframe[(dataframe['target']==1) & (dataframe['ai_pred']==0)]
print("True Positives = " + str(len(true_positives)))

print("True Negatives = " + str(len(true_negatives)))

print("False Positives = " + str(len(false_positives)))

print("False Negatives = " + str(len(false_negatives)))