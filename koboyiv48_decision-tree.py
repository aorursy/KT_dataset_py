!pip install pydotplus
import numpy as np

import pandas as pd 

import glob
# import datasets

files = glob.glob("../input/*.csv")

list = []

for f in files:

    df = pd.read_csv(f,index_col=None)

    list.append(df)

df = pd.concat(list)
# check count missing value

df.isnull().sum()
df[df.to_address.isna()].head()
# drop row missing value from index

index_df = df[df.to_address.isna()].index

df = df.drop(index_df, axis=0)
df.info()
df = df.drop(['from_address','to_address','date'], axis=1)
df.columns
cols = ['value', 'balance', 'open', 'high', 'low', 'close',

       'volumefrom']
X = df[cols]

y = df['status']
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                   test_size=0.3)
# build model

model = DecisionTreeClassifier(criterion='gini')

model.fit(X_train, y_train)
fs = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)

fs
y_train.value_counts()
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO

import pydotplus

from IPython.display import Image

from graphviz import Source
dot_data = StringIO()

export_graphviz(model, out_file = dot_data, 

                feature_names = cols,

                class_names = ['S_low','S_high','low','high'],

                rounded = True, filled = True,

                special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report

print(accuracy_score(y_test, prediction))

print(classification_report(y_test, prediction))