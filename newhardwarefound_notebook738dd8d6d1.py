# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.isnull().values.any()
plt.scatter(df['satisfaction_level'],df['last_evaluation'],c=df['left'])
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth = 10)

clf = clf.fit(df[['satisfaction_level','last_evaluation']], df['left'])
import graphviz

dot_data= tree.export_graphviz(clf, out_file=None,

                              feature_names=['S','E'],  

                              class_names=['L', 'NL'],  

                              filled=True, rounded=True)

graph = graphviz.Source(dot_data)

graph
x = np.linspace(0,1,100)

grid = np.array([[z1,z2] for z1 in x for z2 in x])

response = clf.predict(grid)
plt.scatter(grid[:,0],grid[:,1],c=response)

plt.gca().set_ylim([0.3,1.0])
df.shape
from sklearn import tree

for d in range(1,10):

    clf = tree.DecisionTreeClassifier(max_depth = d)

    clf = clf.fit(df[['satisfaction_level','last_evaluation']], df['left'])

    print("tree depth = {} : {}".format(d,sum(clf.predict(df[['satisfaction_level','last_evaluation']]) != df['left'])/df.shape[0]))