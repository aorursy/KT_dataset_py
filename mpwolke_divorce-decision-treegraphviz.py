# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/divorce-prediction/divorce_data.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'divorce_data.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
x = df.drop(['Q31', 'Divorce'], axis=1)

x.fillna(999999, inplace=True)

y = df['Divorce']
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x, y)
dt_feat = pd.DataFrame(dt.feature_importances_, index=x.columns, columns=['feat_importance'])

dt_feat.sort_values('feat_importance').tail(8).plot.barh()

plt.show()
from IPython.display import SVG

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



graph = Source(export_graphviz(dt, out_file=None, feature_names=x.columns, filled = True))

display(SVG(graph.pipe(format='svg')))