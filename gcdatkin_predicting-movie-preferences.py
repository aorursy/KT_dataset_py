import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
data = pd.read_csv('../input/top-personality-dataset/2018-personality-data.csv')
data
data.info()
data.isna().sum()
data.columns
data = data.drop(['userid',

                  ' movie_1', ' predicted_rating_1',

                  ' movie_2', ' predicted_rating_2',

                  ' movie_3', ' predicted_rating_3',

                  ' movie_4', ' predicted_rating_4',

                  ' movie_5', ' predicted_rating_5',

                  ' movie_6', ' predicted_rating_6',

                  ' movie_7', ' predicted_rating_7',

                  ' movie_8', ' predicted_rating_8',

                  ' movie_9', ' predicted_rating_9',

                  ' movie_10', ' predicted_rating_10',

                  ' movie_11', ' predicted_rating_11',

                  ' movie_12', ' predicted_rating_12',

                  ], axis=1)
data
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
data[' assigned condition'].mode()
condition_ordering = [' low', ' medium', ' default', ' high']
def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df



def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
data = ordinal_encode(data, ' assigned condition', condition_ordering)

data = onehot_encode(data, ' assigned metric', 'm')
data
y = data[' enjoy_watching ']

X = data.drop(' enjoy_watching ', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
log_model = LogisticRegression()

svm_model = SVC(C=1.0)

ann_model = MLPClassifier(hidden_layer_sizes=(16))



log_model.fit(X_train, y_train)

svm_model.fit(X_train, y_train)

ann_model.fit(X_train, y_train)



log_acc = log_model.score(X_test, y_test)

svm_acc = svm_model.score(X_test, y_test)

ann_acc = ann_model.score(X_test, y_test)
fig = px.bar(

    x=['Logistic Regression', 'Support Vector Machine', 'Neural Network'],

    y=[log_acc, svm_acc, ann_acc],

    color=['Logistic Regression', 'Support Vector Machine', 'Neural Network'],

    labels={'x': "Model", 'y': "Accuracy"},

    title="Model Accuracy"

)



fig.show()
1/5
print("Logistic Regression:", log_acc)

print("Support Vector Machine:", svm_acc)

print("Neural Network:", ann_acc)