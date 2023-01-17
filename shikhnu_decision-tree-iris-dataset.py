#Loading Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import sklearn.datasets as datasets

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.tree import plot_tree



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



from sklearn.metrics import confusion_matrix 

from sklearn.metrics import classification_report 
# Loading the iris dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df = pd.read_csv(url, header=None, names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',

                                          'petal width (cm)', 'Species'])



df.head() # To view first 5 rows
# To know number of rows and collumns

df.shape
# Check the dataframe information

df.info()
# To find if any null value is present

df.isnull().sum()
# To see summary statistics

df.describe().T
# To find outliers

cols = df.columns[0:-1]

for i in cols:

    sns.boxplot(y=df[i])

    plt.show()
# To remove outliers from 'sepal width (cm)'

q1 = df['sepal width (cm)'].quantile(0.25)

q3 = df['sepal width (cm)'].quantile(0.75)

iqr = q3 - q1

df = df[(df['sepal width (cm)'] >= q1-1.5*iqr) & (df['sepal width (cm)'] <= q3+1.5*iqr)]

df.shape # To find out the number of rows and column after outlier treatment
# Blocplot for sepal width (cm) after outlier treatment

sns.boxplot(y=df['sepal width (cm)'])

plt.show()
# Splitting the data into train and test sets

X = df.drop("Species",axis=1)

y = df["Species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state= 1)
# Defining an object for DTC and fitting for whole dataset

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1 )

dt.fit(X, y)
# Plotting of decission tree

from IPython.display import Image

from sklearn.tree import export_graphviz



!pip install pydotplus

import pydotplus





features = X.columns

dot_data = export_graphviz(dt, out_file=None, feature_names=features)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
# Defining an object for DTC and fitting for train dataset

dt = DecisionTreeClassifier(random_state=1)

dt.fit(X_train, y_train)



y_pred_train = dt.predict(X_train)

y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)
print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))
#Classification for test before hyperparameter tuning

print(classification_report(y_test,y_pred))
# Hyperparameter Tuning of DTC



dt = DecisionTreeClassifier(random_state=1)



params = {'max_depth' : [2,3,4,5],

        'min_samples_split': [2,3,4,5],

        'min_samples_leaf': [1,2,3,4,5]}



gsearch = GridSearchCV(dt, param_grid=params, cv=3)



gsearch.fit(X,y)



gsearch.best_params_
# Passing best parameter for the Hyperparameter Tuning

dt = DecisionTreeClassifier(**gsearch.best_params_, random_state=1)



dt.fit(X_train, y_train)



y_pred_train = dt.predict(X_train)

y_prob_train = dt.predict_proba(X_train)[:,1]



y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)[:,1]
print('Confusion Matrix - Train:','\n',confusion_matrix(y_train,y_pred_train))

print('\n','Confusion Matrix - Test:','\n',confusion_matrix(y_test,y_pred))
#Classification for test after hyperparameter tuning

print(classification_report(y_test,y_pred))
print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))