# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/pointcloudmnist2d/train.csv")

df_test = pd.read_csv("/kaggle/input/pointcloudmnist2d/test.csv")



X = df_train[df_train.columns[1:]].to_numpy()

y = df_train[df_train.columns[0]].to_numpy()



X = X.reshape(X.shape[0], -1, 3)



X.shape, y.shape
for length in [50, 100, 200, 351]:

    print(f"Sample {length}")

    

    idx = np.random.choice(len(X), 3)

    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"Number {label}" for label in y[idx]])



    for col, idx in enumerate(idx):

        Xi = X[idx]

        yi = y[idx]



        Xi = Xi[:length]

        Xi = Xi[Xi[:, 2] > 0]





        fig.add_trace(

            go.Scatter(

                x=Xi[:, 0],

                y=Xi[:, 1],

                mode="markers",

                marker=dict(

                    size=Xi[:, 2] / 16.0,

                    color="black",

                )

            ),

            row=1,

            col=col + 1,

        )

        

        fig.update_xaxes(range=[-1, 28], row=1, col=col+1)

        fig.update_yaxes(range=[-1, 28], row=1, col=col+1)



    fig.show()