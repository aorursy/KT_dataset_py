# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/dec.csv")
features = data[['Outlook','Temp','Humidity','Wind']]

df_outlook = pd.get_dummies(data[['Outlook']],drop_first=True)

df_Temp = pd.get_dummies(data[['Temp']],drop_first=True)

df_Humidity = pd.get_dummies(data[['Humidity']],drop_first=True)

df_wind = pd.get_dummies(data[['Wind']],drop_first=True)

finalDf = pd.concat([df_outlook,df_Temp,df_Humidity,df_wind],axis=1)

res=[]

for i in data['Decision']:

    res.append(i)

finalDf.columns
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(criterion = "entropy")

dtree.fit(finalDf,res)

dtree.predict([[0,0,0,0,0,0]])
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydot

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,feature_names  =finalDf.columns,

                filled=True, rounded=True,

                special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png())