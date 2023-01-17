# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import plotly.express as px

import plotly.graph_objs as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")
df
df.isnull().sum()
df=df.dropna()

df.isnull().sum()
X_30 = df.drop('TenYearCHD', axis=1).to_numpy()

y_text = df['TenYearCHD'].to_numpy()
X_30
X_30.shape
y_text
y_text.shape
pca = PCA(n_components=2)

pca.fit(X_30)

X = pca.transform(X_30)
X
X.shape
df = pd.DataFrame(data=np.c_[X, y_text], columns=['Feature 1', 'Feature 2', 'Label'])
df
fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Label')

fig.show()
y = (2 * LabelEncoder().fit_transform(y_text)) - 1
y
y.shape
points_colorscale = [

                     [0.0, 'rgb(239, 85, 59)'],

                     [1.0, 'rgb(99, 110, 250)'],

                    ]



layout = go.Layout(scene=dict(

                              xaxis=dict(title='Feature 1'),

                              yaxis=dict(title='Featrue 2'),

                              zaxis=dict(title='Label')

                             ),

                  )



points = go.Scatter3d(x=df['Feature 1'], 

                      y=df['Feature 2'], 

                      z=y,

                      mode='markers',

                      text=df['Label'],

                      marker=dict(

                                  size=3,

                                  color=y,

                                  colorscale=points_colorscale

                            ),

                     )



fig2 = go.Figure(data=[points], layout=layout)

fig2.show()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
detail_steps = 100



(x_vis_0_min, x_vis_1_min) = X_train.min(axis=0)

(x_vis_0_max, x_vis_1_max) = X_train.max(axis=0)



x_vis_0_range = np.linspace(x_vis_0_min, x_vis_0_max, detail_steps)

x_vis_1_range = np.linspace(x_vis_1_min, x_vis_1_max, detail_steps)



(XX_vis_0, XX_vis_1) = np.meshgrid(x_vis_0_range, x_vis_0_range)



X_vis = np.c_[XX_vis_0.reshape(-1), XX_vis_1.reshape(-1)]
probs=logreg.predict_proba(X_vis)
probs.shape
yhat_vis = (2 * probs[:, 1]) - 1
YYhat_vis = yhat_vis.reshape(XX_vis_0.shape)



surface_colorscale = [

                      [0.0, 'rgb(235, 185, 177)'],

                      [1.0, 'rgb(199, 204, 249)'],

                     ]



surface = go.Surface(

                     x=XX_vis_0, 

                     y=XX_vis_1,

                     z=YYhat_vis,

                     colorscale=surface_colorscale,

                     showscale=False

                    )



fig3 = go.Figure(data=[points, surface], layout=layout)

fig3.show()
yhat_train=logreg.predict(X_train)
accuracy_score(yhat_train, y_train)
yhat_test=logreg.predict(X_test)
accuracy_score(yhat_test, y_test)