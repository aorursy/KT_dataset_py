import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



from sklearn.metrics import roc_auc_score
data = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data
data.info()
data.isna().sum()
data = data.drop('PassengerId', axis=1)
data['Category'].unique()
data['Country'].unique()
data['Lastname'] = data['Lastname'].apply(lambda x: x[0])

data = data.drop('Firstname', axis=1)
data
def binary_encode(df, column, postive_value):

    df = df.copy()

    df[column] = df[column].apply(lambda x: 1 if x == postive_value else 0)

    return df



def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df

    

def onehot_encode(df, column):

    df = df.copy()

    dummies = pd.get_dummies(df[column])

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
alphabet_ordering = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
data = binary_encode(data, 'Sex', 'M')

data = binary_encode(data, 'Category', 'M')



data = ordinal_encode(data, 'Lastname', alphabet_ordering)



data = onehot_encode(data, 'Country')
data
y = data['Survived']

X = data.drop('Survived', axis=1)
scaler = MinMaxScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
log_model = LogisticRegression()

svm_model = SVC(C=1.0)

ann_model = MLPClassifier(hidden_layer_sizes=(16, 16))



log_model.fit(X_train, y_train)

svm_model.fit(X_train, y_train)

ann_model.fit(X_train, y_train)
log_acc = log_model.score(X_test, y_test)

svm_acc = svm_model.score(X_test, y_test)

ann_acc = ann_model.score(X_test, y_test)



log_preds = log_model.predict(X_test)

svm_preds = svm_model.predict(X_test)

ann_preds = ann_model.predict(X_test)



log_auc = roc_auc_score(y_test, log_preds)

svm_auc = roc_auc_score(y_test, svm_preds)

ann_auc = roc_auc_score(y_test, ann_preds)
acc_fig = px.bar(

    x = ["Logistic Regression", "Support Vector Machine", "Neural Network"],

    y = [log_acc, svm_acc, ann_acc],

    labels={'x': "Model", 'y': "Accuracy"},

    color=["Logistic Regression", "Support Vector Machine", "Neural Network"],

    title="Model Accuracy"

)



acc_fig.show()
1 - (y.sum() / len(y))
auc_fig = px.bar(

    x = ["Logistic Regression", "Support Vector Machine", "Neural Network"],

    y = [log_auc, svm_auc, ann_auc],

    labels={'x': "Model", 'y': "ROC AUC"},

    color=["Logistic Regression", "Support Vector Machine", "Neural Network"],

    title="Model ROC AUC"

)



auc_fig.show()