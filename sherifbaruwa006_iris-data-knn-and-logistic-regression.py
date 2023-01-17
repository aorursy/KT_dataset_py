import seaborn as sns; sns.set(style= "darkgrid", color_codes = True)

sns.set(rc={'figure.figsize':(11.7,8.27)})

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

plt.rcParams['figure.figsize'] = (12.0, 9.0)

import warnings; warnings.filterwarnings('ignore')
dataset = pd.read_csv("../input/iris.csv")
dataset.head()
dataset.info()
dataset.describe()
dataset.isnull().sum()
sns.pairplot(dataset, hue = 'species', height= 5, palette = "Set2")

plt.show()
sns.heatmap(dataset.corr(), annot= True, fmt= '.2g', cmap= 'Set2')
dataset.shape
ind = np.arange(150)

np.random.seed(1)

np.random.shuffle(ind)

iris_data = dataset.iloc[ind]
iris_data.shape
iris_data.head(5)
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(iris_data.drop('species',axis = 1)) 
scale = scaler.transform(iris_data.drop('species',axis = 1)) 
iris_scaled = pd.DataFrame(scale, columns= iris_data.columns[:-1])
iris_scaled.head()
from sklearn.model_selection import train_test_split
X = iris_scaled



Y = iris_data['species']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train, y_train)



knn_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
knn_cm = pd.DataFrame(confusion_matrix(y_test, knn_pred))
print(knn_cm, "\n")

print(classification_report(y_test, knn_pred))
knn_acc = accuracy_score(y_test, knn_pred)

knn_acc
error_rate = []
for i in range(1,11):

    knn = KNeighborsClassifier(i)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    error_rate.append(np.mean(pred != y_test))
plt.plot(error_rate, 'b--', marker = 'o', markerfacecolor = "red")
knn = KNeighborsClassifier(n_neighbors=6)



knn.fit(X_train, y_train)



knn_pred2 = knn.predict(X_test)
knn_cm2 = pd.DataFrame(confusion_matrix(y_test, knn_pred2))



sns.heatmap(confusion_matrix(y_test, knn_pred2), annot= True, fmt= 'd',

            xticklabels= ['setosa', 'versicolor', 'virginica'], 

            yticklabels= ['setosa', 'versicolor', 'virginica'], 

            cmap= "Set2")
print('with K = 6 \n')

print(classification_report(y_test, knn_pred2))
knn_acc2 = accuracy_score(y_test, knn_pred2)

knn_acc2
from sklearn.linear_model import LogisticRegression
iris_data.head(3)
iris_data.shape
X = iris_data.drop('species', axis= 1)



Y = iris_data['species']



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
glm = LogisticRegression()



glm.fit(X_train, y_train)



glm_pred = glm.predict(X_test)
sns.heatmap(confusion_matrix(y_test, glm_pred), annot= True, fmt= 'd',

            xticklabels= ['setosa', 'versicolor', 'virginica'], 

            yticklabels= ['setosa', 'versicolor', 'virginica'], 

            cmap= "Set2")
print(classification_report(y_test, glm_pred))
glm_acc = accuracy_score(y_test, glm_pred)

glm_acc
scores = [glm_acc, knn_acc2]
algorithms = ["Logistic Regression","K Nearest Neighbor"]
sns.barplot(algorithms,scores, palette= 'Set2')