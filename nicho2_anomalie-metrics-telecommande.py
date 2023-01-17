# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from IPython.display import Markdown # pour affichage dans le Markdown

from pandas.io.json import json_normalize #package for flattening json in pandas df

import json 

#load json object

with open('../input/historique_status_panneau_lighting.json') as f:

    d = json.load(f)

    

Markdown( f"""# Nombre d'enregistrements : {len(d)}""")
df = json_normalize(data=d,record_path="values")

df.loc[:,'date'] = pd.to_datetime(df['date'])

df['diff'] = (df['date'] - df['date'].shift(-1))

df['diff_en_s'] = df['diff'].dt.total_seconds()

df.head()
df.describe()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go







import numpy as np





x = df.diff_en_s.dropna()

layout = go.Layout(

        title=go.layout.Title(text="répartition des échantillons")

    )

fig = go.Figure(data=[go.Histogram(x=x)],layout=layout)



fig.show()
val = df.loc[(df['diff_en_s']> (12*3600))].diff_en_s.count()

Markdown( f"""# nombre de fois où le temps entre 2 métriques a dépassé 12h ---->  {val}

 # donc {val*100/len(df)} % du nombre total de métriques""")
df.info()