import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/iris/Iris.csv')
data.head()
data.describe()
tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')
plt.show()
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
k_range = list(range(1,26))
scores = []
models = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    models.append(knn)
    
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()
print(np.argmax(scores[5:])+1, 'neigbours', max(scores[5:]))
pd.DataFrame(metrics.classification_report(knn.predict(X_test), y_test, output_dict=True))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
pd.DataFrame(metrics.classification_report(nb.predict(X_test), y_test, output_dict=True))