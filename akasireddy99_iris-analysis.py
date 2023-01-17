import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



dataset = pd.read_csv("../input/IRIS.csv")



dataset.head()
dataset['species'].value_counts()
fig = dataset[dataset['species'] == 'Iris-setosa'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='orange', label='Setosa')

dataset[dataset['species'] == 'Iris-versicolor'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='yellow', label='Versicolor', ax=fig)

dataset[dataset['species'] == 'Iris-virginica'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='blue', label='Verginica', ax=fig)

fig.set_ylabel('Sepal Width')

fig.set_xlabel('Sepal Length')

fig.set_title('Sepal Length vs Width')



fig = plt.gcf()

fig.set_size_inches(18, 9)

plt.show()
dataset['species'].groupby(dataset['petal_width']).value_counts()
dataset['species'].groupby(pd.qcut(dataset['petal_width'], 3)).value_counts()
dataset['petal_area'] = dataset.apply(lambda row: (row['petal_length'] * row['petal_width']), axis=1)
dataset.head()
dataset['species'].groupby(pd.qcut(dataset['petal_area'], 3)).value_counts()
dataset['sepal_area'] = dataset.apply(lambda row: (row['sepal_length'] * row['sepal_width']), axis=1)
dataset.head()
dataset['species'].groupby(pd.qcut(dataset['sepal_area'], 3)).value_counts()
sns.lineplot(dataset['petal_area'], dataset['sepal_area'])
dataset['area_diff'] = dataset.apply(lambda row: (row['sepal_area'] - row['petal_area']), axis=1)
dataset.head()
dataset['species'].groupby(pd.qcut(dataset['area_diff'], 3)).value_counts()
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

dataset['species'] = label_encoder.fit_transform(dataset['species'])



rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=1000,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(dataset.iloc[:, 5:-1], dataset.iloc[:, 4])

print("%.4f" % rf.oob_score_)