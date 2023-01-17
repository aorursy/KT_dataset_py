import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/Iris.csv")
df.head()
df.info()
df.describe()
iris_type = df['Species']
df_noID = df.drop('Id', axis = 1)
df_noID.head()
sns.pairplot(df_noID, hue='Species')
y = df_noID['Species'].values   
X = df_noID.drop('Species', axis=1).values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
neighbors = np.arange(1, 30)
score_iter = []
score = []
for k in neighbors:
    knn_iter = KNeighborsClassifier(n_neighbors=k)
    knn_iter.fit(X_train, y_train)
    score_iter= knn_iter.score(X_test, y_test)
    score.append(score_iter)
plt.plot(neighbors, score)
plt.title('Number of neighbors vs. score')
plt.xlabel('Number of neighbors')
plt.ylabel('Model score')
plt.show()

