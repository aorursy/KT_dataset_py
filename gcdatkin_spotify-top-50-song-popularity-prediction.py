import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
data = pd.read_csv('../input/top50spotify2019/top50.csv', encoding='latin-1')
data
data.info()
data = data.drop(['Unnamed: 0', 'Track.Name'], axis=1)
data
data['Popularity'] = pd.qcut(data['Popularity'], q=2, labels=[0, 1])
def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
data = onehot_encode(data, 'Genre', 'genre')

data = onehot_encode(data, 'Artist.Name', 'artist')
data
y = data.loc[:, 'Popularity']

X = data.drop('Popularity', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=20)
log_model = LogisticRegression()

knn_model = KNeighborsClassifier()

dec_model = DecisionTreeClassifier()

mlp_model = MLPClassifier()

svm_model = SVC()
log_model.fit(X_train, y_train)

knn_model.fit(X_train, y_train)

dec_model.fit(X_train, y_train)

mlp_model.fit(X_train, y_train)

svm_model.fit(X_train, y_train)
log_acc = log_model.score(X_test, y_test)

knn_acc = knn_model.score(X_test, y_test)

dec_acc = dec_model.score(X_test, y_test)

mlp_acc = mlp_model.score(X_test, y_test)

svm_acc = svm_model.score(X_test, y_test)
print("Logistic Regression Accuracy:", log_acc)

print("K-Nearest-Neighbors Accuracy:", knn_acc)

print("Decision Tree Accuracy:", dec_acc)

print("Neural Network Accuracy:", mlp_acc)

print("Support Vector Machine Accuracy:", svm_acc)
fig = px.bar(

    x=["Logistic Regression", "K-Nearest-Neighbors", "Decision Tree", "Neural Network", "Support Vector Machine"],

    y=[log_acc, knn_acc, dec_acc, mlp_acc, svm_acc],

    color=["Logistic Regression", "K-Nearest-Neighbors", "Decision Tree", "Neural Network", "Support Vector Machine"],

    labels={'x': "Model", 'y': "Accuracy"},

    title="Model Accuracy Comparison"

)



fig.show()