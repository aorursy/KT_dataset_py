import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df['class'].unique()
df.isnull().sum()
df.shape
df.info()
plt.figure(figsize=(7,7))

sns.countplot('class', data=df)

plt.show()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder() 

for col in df.columns: 

    df[col]=labelencoder.fit_transform(df[col])

    

df.head()
plt.figure(figsize=(15,7))

sns.boxplot(df)

plt.show()
X = df.iloc[:, 1:]

y = df.iloc[:, 0]
X.head()
y.head()
X.describe()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), annot=True, cmap='seismic_r', linewidths=.5)

plt.show()
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()

X = scaler.fit_transform(X)

X
from sklearn.decomposition import PCA

N = df.values

pca = PCA(n_components=2)

x = pca.fit_transform(N)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=5)

X_clustered = kmeans.fit_predict(N)



LABEL_COLOR_MAP = {0 : 'g',

                   1 : 'y',

                   2 : 'r',

                   3 : 'b'

                  }



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (15,7))

plt.scatter(x[:,0],x[:,1], c= label_color)

plt.show()
X_clustered
pca=PCA(n_components=20)



X = pca.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
logreg = LogisticRegression() 

logreg.fit(X_train, y_train)

log_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

gnb_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, y_train)

per_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

lin_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

dec_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

ran_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)