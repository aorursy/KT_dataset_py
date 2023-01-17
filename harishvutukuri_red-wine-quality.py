import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv("../input/winequality-red.csv")
dataset.head()
dataset.shape
dataset.isna().sum()
# Univariate graphs to see the distribution
dataset.hist(figsize=(20, 15))
plt.show()
# Correlation Matrix
plt.subplots(figsize=(20, 15))
sns.heatmap(dataset.corr(), annot=True)
bins = (2, 6, 8)
group_names = ['bad', 'good']
dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
from sklearn.preprocessing import LabelEncoder
label_quality = LabelEncoder()
dataset['quality'] = label_quality.fit_transform(dataset['quality'])
dataset["quality"].value_counts()
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
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
results
accuracy
# Parameter Tuning the Best Model from the results
from sklearn.model_selection import GridSearchCV

model = ExtraTreesClassifier(random_state=seed)

params = {'n_estimators':list(range(90,100)), 'criterion':['gini','entropy']}

grid_search = GridSearchCV(estimator = model ,param_grid = params,scoring=scoring ,cv =num_folds, verbose = 4) 
grid_search.fit(X_train, y_train)
# Best Score and Best Parameters from GridSearch
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
# Finalizing the model and comparing the test, predict results

model = ExtraTreesClassifier(random_state=seed, n_estimators = 91, criterion='gini')

_ = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

accuracy["GScv"] = accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))

cm= confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
accuracy