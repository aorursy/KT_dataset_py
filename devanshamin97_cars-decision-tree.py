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
from sklearn import tree

df = pd.read_csv('/kaggle/input/carsdata/cars.csv', na_values =' ')

df.head()
df.columns =['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',

       'year', 'brand']

df = df.dropna()

df.head()
d = {' US.' : 0, ' Japan.' : 1, ' Europe.' : 2}

df['brand'] = df['brand'].map(d)

df.head() 
features = pd.get_dummies(df[ ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year'] ])

features
clf = tree.DecisionTreeClassifier()

clf = clf.fit(features,df['brand'])
#pip install pydotplus
# To display decision tree



#from IPython.display import Image  

#from sklearn.externals.six import StringIO  

#import pydotplus



#dot_data = StringIO()  

#tree.export_graphviz(clf, out_file=dot_data,  

#                         feature_names=list(features.columns.values))  

#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#Image(graph.create_png())
# Using random forest to make sure my model doesn't overfit



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 20) #n_esitmators value can be changed according to your need

clf = clf.fit(features,df['brand'])

print(clf.predict([[14.0,8,350,165,4209,12,1972]]))

print(clf.predict([[31.9,4,89,71,1925,14,1980]]))