# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
column=['party', 'infants', 'water', 'budget', 'physician', 'salvador',

       'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',

       'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']

df=pd.read_csv('../input/house-votes-84.csv')

df.head()
df.columns=column
df.head()
df.dropna(how='all')

df.head()

df.replace('n',0,inplace=True)

df.replace('y',1,inplace=True)

df.replace('?',np.nan,inplace=True)

df.head()
# Import the Imputer module

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import classification_report 



# Setup the Imputation transformer: imp

imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)



# Instantiate the SVC classifier: clf

clf = SVC()



# Setup the pipeline with the required steps: steps

steps = [('imputation', imp),

        ('SVM', clf)]



pipeline = Pipeline(steps)

pipeline = Pipeline(steps)



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)



# Fit the pipeline to the train set

pipeline.fit(X_train,y_train)



# Predict the labels of the test set

y_pred = pipeline.predict(X_test)



# Compute metrics

print(classification_report(y_test,y_pred))
plt.figure()

sns.countplot(x='education', hue='party', data=df, palette='RdBu')

plt.xticks([0,1], ['No', 'Yes'])

plt.show()
sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')

plt.show()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')

plt.show()
df=df.replace(to_replace=['n', 'y',np.nan], value=[0, 1, 0])

df.head()


# Import KNeighborsClassifier from sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier



# Create arrays for the features and the response variable

y = df['party'].values

X = df.drop('party', axis=1).values



# Create a k-NN classifier with 6 neighbors

knn = KNeighborsClassifier(n_neighbors=6)



# Fit the classifier to the data

knn.fit(X_train,y_train)
y = df['party'].values

X = df.drop('party', axis=1).values
knn.fit(X, y)
y_pred = knn.predict(X)



# Predict and print the label for the new data point X_new

new_prediction = knn.predict(X)

print("Prediction: {}".format(new_prediction)) 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

logreg = LogisticRegression(solver='lbfgs')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
from sklearn.metrics import roc_curve

y_pred_prob = logreg.predict_proba(X_test)[:,1]

# print(y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='T')
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC Curve')

plt.show()           
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 20)}

knn_gs = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_gs, param_grid, cv=5)

knn_cv.fit(X, y)
knn_cv.best_params_
knn_cv.best_score_
# Import necessary modules

from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)



# Fit it to the data

tree_cv.fit(X,y)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))

# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Create the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}



# Instantiate the logistic regression classifier: logreg

logreg = LogisticRegression(solver='liblinear')



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg,param_grid,cv=5)



# Fit it to the training data

logreg_cv.fit(X_train,y_train)



# Print the optimal parameters and best score

print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))

print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
