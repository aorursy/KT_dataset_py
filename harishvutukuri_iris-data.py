# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # Importing metrics for evaluation
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.simplefilter(action='ignore')
# importing the dataset
dataset = pd.read_csv("../input/Iris.csv")
dataset.head()
dataset.shape
# visualize data for correlation and correlation matrix
sns.pairplot(dataset, hue='Species')
dataset.Species.value_counts()
# correlation between the variables
plt.figure(figsize=(20,10)) 
sns.heatmap(dataset.corr(),annot=True)
dataset['Species'].replace("Iris-setosa",1,inplace= True)
dataset['Species'].replace("Iris-virginica",2,inplace = True)
dataset['Species'].replace("Iris-versicolor",3,inplace=True)
# Data wrangling
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Test options and evaluation metric
num_folds = 10
seed = 0
scoring = 'accuracy'
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Spot-Check Algorithms (Classification)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Spot-Check Ensemble Models (Classification)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB',XGBClassifier()))

# evaluate each model in turn
results = {}
accuracy = {}
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results[name] = (cv_results.mean(), cv_results.std())
    model.fit(X_train, y_train)
    _ = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, _)
accuracy