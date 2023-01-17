# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/if-covid19-surveillance-data-set/Surveillance.csv')
df.head()
x = df.drop('Categories',axis = 'columns')
y = df['Categories']
y
from sklearn.preprocessing import LabelEncoder #Machine Learning hanya memahami number
le_A01 = LabelEncoder()
le_A02 = LabelEncoder()
le_A03 = LabelEncoder()
le_A04 = LabelEncoder()
le_A05 = LabelEncoder()
le_A06 = LabelEncoder()
le_A07 = LabelEncoder()
x['A01_n'] = le_A01.fit_transform(x['A01'])
x['A02_n'] = le_A02.fit_transform(x['A02'])
x['A03_n'] = le_A03.fit_transform(x['A03'])
x['A04_n'] = le_A04.fit_transform(x['A04'])
x['A05_n'] = le_A05.fit_transform(x['A05'])
x['A06_n'] = le_A06.fit_transform(x['A06'])
x['A07_n'] = le_A07.fit_transform(x['A07'])
x.head()
x_n = x.drop(['A01','A02','A03','A04','A05','A06','A07'],axis = 'columns')
x_n
from sklearn import tree
pohon = tree.DecisionTreeClassifier()
pohon.fit(x_n,y)
tree.plot_tree(pohon)
import graphviz
dot_data = tree.export_graphviz(pohon, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("if-covid19-surveillance-data-set")

dot_data = tree.export_graphviz(pohon, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)  
graph 
from sklearn.tree import export_text
rara = export_text(pohon)
print (rara)
