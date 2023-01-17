import pandas as pd

import numpy as np



import plotly.express as px

import plotly.graph_objects as go



import missingno as msno
df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding = "ISO-8859-1")
!pip install pydotplus
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split





import pydotplus

from IPython.display import display

#from IPython import display

from IPython.display import Image

from graphviz import Digraph
df.head()
#列名を扱いやすいようにする



df.rename(columns={

    "Unnamed: 0":"ID"

}, inplace=True)





df.rename(columns=lambda x : str(x).replace(".",""), inplace=True)



df.set_index("ID", inplace=True)
df.reset_index().drop(columns=["ID","TrackName"], inplace=True)
df_get_dummy = pd.get_dummies(df)

df_get_dummy.head()
#曲の定量数値のみ入れてみる



Y_2 = df_get_dummy["Popularity"]

X_2 = df.drop(columns=["TrackName","ArtistName","Genre","Popularity"])
#学習

clf_2 = tree.DecisionTreeRegressor(max_depth=2)

clf_2.fit(X_2,Y_2)
#可視化

dot_data_2 = tree.export_graphviz(

    clf_2,

    out_file=None,

    feature_names=X_2.columns,

    filled=True,

    proportion=False,

    rounded=True)



graph_2 = pydotplus.graph_from_dot_data(dot_data_2)

Image(graph_2.create_png())
#!pip install graphviz

!pip install dtreeviz
import urllib.request

import dtreeviz
viz = dtreeviz(

    clf_2,

    np.array(X_2), 

    np.array(Y_2),

    target_name='Popularity',

    feature_names=X_2.columns,

    ) 
X_2.columns