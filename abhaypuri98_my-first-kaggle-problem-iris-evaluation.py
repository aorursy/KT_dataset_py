import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")

iris=pd.read_csv('../input/Iris.csv')
iris.head()
iris.info()
iris.tail()
iris1=iris.drop("Id", axis=1)
iris1.columns= ['sepal_length', 'sepal_width' , 'petal_length', 'petal_width','species']
iris1.head()
iris1.sample(3)
iris1['Species']=iris['Species'].astype('category')
iris1.dtypes
print(iris1.Species.unique())
print(iris1['Species'].value_counts())
iris2=pd.DataFrame(iris['Species'].value_counts())
iris2
iris1.shape
iris1.describe()
iris1.size
iris1.isnull().sum()
iris1.min()
iris1.max()
iris1.median()
iris1['Species'].value_counts().plot(kind="bar");
sns.set(style="whitegrid", palette="GnBu_d", rc={'figure.figsize':(11.7,8.27)})

title="Compare the Distributions of Sepal Length"

sns.boxplot(x="species", y="sepal_length", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()
title="Compare the Distributions of Sepal Width"

sns.boxplot(x="species", y="sepal_width", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()
title="Compare the Distributions of Petal Length"

sns.boxplot(x="species", y="petal_length", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()
title="Compare the Distributions of Petal width"

sns.boxplot(x="species", y="petal_width", data=iris1)

# increasing font size
plt.title(title, fontsize=26)
# Show the plot
plt.show()
sns.countplot(x='petal_length', data = iris1)
sns.countplot(x='petal_width', data = iris1)
plt.figure(figsize=(7,4)) 
sns.heatmap(iris1.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()
corr = iris1.corr()
corr
# import correlation matrix to see parametrs which best correlate each other
# According to the correlation matrix results Petal LengthCm and
#PetalWidthCm have positive correlation which is proved by the scatter plot discussed above

import seaborn as sns
import pandas as pd
corr = iris1.corr()
plt.figure(figsize=(10,8)) 
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap='viridis', annot=True)
plt.show()
# Modify the graph above by assigning each species an individual color.
sns.FacetGrid(iris1, hue="Species", size=5) \
   .map(plt.scatter, "sepal_length", "sepal_width") \
   .add_legend()
plt.show()
X = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
logreg = LogisticRegression()
logreg.fit(X, y)
y_pred = logreg.predict(X)
print(metrics.accuracy_score(y, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

Model = GaussianNB()
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

Model = DecisionTreeClassifier()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
Model=RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
