# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set(style="ticks", color_codes=True)
df = pd.read_csv('../input/iris/Iris.csv')
df.head(10)
df.drop('Id', axis=1, inplace=True)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df.head(10)
print(f'Number of Lines: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
df.info()
df.describe()
df['species'].value_counts()
ax = sns.pairplot(df)
ax = sns.pairplot(df, hue='species')
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(f'X: {X_train.shape}')
print(f'y: {X_test.shape}')
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)
# Testing model
X_new = np.array([[5, 2.9, 1, 0.2]])
X_new.shape
prediction = model.predict(X_new)
prediction
# Accuracy of the model
model.score(X_test, y_test)