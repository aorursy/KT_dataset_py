# Import the usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
print(pd.__version__, np.__version__)
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')

df = pd.concat([train_df, test_df], sort=True)
df.sample(10)
df[['Age', 'Sex']].isnull().sum()
df['Age'].describe()
# Quantity of people by given age
max_age = df['Age'].max()
df['Age'].hist(bins=int(max_age))
# Survival ratio per decade, ignoring NaN with dropna()
df['decade'] = df['Age'].dropna().apply(lambda x: int(x/10))
df[['decade', 'Survived']].groupby('decade').mean().plot()
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
df['male'] = df['Sex'].map({'male': 1, 'female': 0})
df.sample(5)
df[['male','Survived']].groupby('male').mean()
train = df[df['Survived'].notnull()]

features = ['Age', 'male']
train_X = train[features]
train_y = train['Survived']
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(train_X, train_y)
from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(classifier, out_file=None,
                           feature_names=features,
                           class_names=['Dead', 'Alive'],  
                           filled=True, rounded=True)
graphviz.Source(dot_data)
classifier.score(train_X, train_y)
test = df[df['Survived'].isnull()]

test_X = test[features]
test_y = classifier.predict(test_X)
submit = pd.DataFrame(test_y.astype(int),
                      index=test_X.index,
                      columns=['Survived'])
submit.head()
submit.to_csv('submit.csv')