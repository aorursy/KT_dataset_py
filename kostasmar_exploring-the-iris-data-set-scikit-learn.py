import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', None)
from sklearn.datasets import load_iris
iris_data = load_iris()
type(iris_data)
iris_data.keys()
iris_data['feature_names']
iris_data['target_names']
iris_df = pd.DataFrame(data = iris_data['data'], columns = iris_data['feature_names'])
iris_df.head()
iris_df['Iris type'] = iris_data['target']
iris_df.head()
iris_df['Iris name'] = iris_df['Iris type'].apply(lambda x: 'sentosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))
iris_df.head()
def f(x):
    if x == 0:
        val = 'setosa'
    elif x == 1:
        val = 'versicolor'
    else:
        val = 'virginica'
    return val
iris_df['test'] = iris_df['Iris type'].apply(f)
iris_df.head()
iris_df.drop(['test'], axis =1, inplace = True)
iris_df.info()
iris_df.describe()
iris_df.groupby(['Iris name']).describe()
iris_df.columns
# im just making a function in order not to repeat the same code (boring)
def plot_violin(y2,i):
    plt.subplot(2,2,i)
    sns.violinplot(x='Iris name',y= y2, data=iris_df)
plt.figure(figsize=(15,10))
i = 1
for measurement in iris_df.columns[:-2]:
    plot_violin(measurement,i)
    i += 1
sns.pairplot(iris_df, hue = 'Iris name', vars = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], palette = 'Set1' )
iris_df.iloc[:,:4].corr()
sns.heatmap(iris_df.iloc[:,:4].corr(), annot = True)
from sklearn.model_selection import train_test_split
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
y = iris_df['Iris name']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred
knn.score(X_test, y_test)