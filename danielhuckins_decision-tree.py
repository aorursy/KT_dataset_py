import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('../input/train.csv')

colors = data['color']

data = data.drop(['color'], axis=1)

data = data.drop(['id'], axis=1)

feature_names = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color']
le = LabelEncoder()

data['color'] = le.fit_transform(colors)
X = data[feature_names]

y = data['type']
cls = DecisionTreeClassifier()

cls.fit(X, y)
test = pd.read_csv('../input/test.csv')
test