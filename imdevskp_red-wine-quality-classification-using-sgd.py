import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")



from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV



from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv('../input/winequality-red.csv')

df.head()
df.describe()
df.info()
qty_cnt = df['quality'].value_counts().sort_index()

qty_cnt.plot(kind='bar')
plt.figure(figsize=(10, 8))

sns.pairplot(df, hue="quality", palette="husl")

plt.plot()
plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(), cmap='RdBu', center=0, annot=True)

plt.plot()
for i in df.columns[:-1]:

    plt.figure()

    sns.boxplot(x=df['quality'], y=df[i])

    plt.plot()
for i in df.columns[:-1]:

    plt.figure()

    sns.barplot(x=df['quality'], y=df[i])

    plt.plot()
'''for i in df.columns[:-1]:

    plt.figure()

    for qn in range(3, 9):

        ax = sns.kdeplot(df[df['quality']==qn][i], shade=True)

        ax.set_label(s=qn)

    plt.plot()'''
df.isna().sum()
bins = (2, 5, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], 

                       bins = bins, 

                       labels = group_names)
enc = LabelEncoder()

df['quality'] = enc.fit_transform(df['quality'])

sns.countplot(df['quality'])
mm = StandardScaler()

for i in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']:

    df[i] =  mm.fit_transform(df[i].values.reshape(-1,1))
X = df.drop(['quality'], axis=1)

y = df['quality']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nbc = GaussianNB()

nbc.fit(X_train, y_train)

nbc_pred = nbc.predict(X_test)



scores = cross_val_score(nbc, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(nbc_pred, y_test))
lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



scores = cross_val_score(lr, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(lr_pred, y_test))
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)



scores = cross_val_score(dt, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(lr_pred, y_test))
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier()



hyperparameters = {

    "n_neighbors": range(1,50,2)

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



best_model = grid.best_estimator_

pred = best_model.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)

print(confusion_matrix(pred, y_test))
model = RandomForestClassifier()



hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   "min_samples_leaf": [1, 5],

                   "min_samples_split": [3, 5],

                   "n_estimators": [6, 9]

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



best_model = grid.best_estimator_

pred = best_model.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)

print(confusion_matrix(pred, y_test))
sgd = SGDClassifier(penalty=None)

sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)



print(accuracy_score(y_test, pred_sgd))

print(confusion_matrix(y_test, pred_sgd))