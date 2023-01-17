import pandas as pd
import numpy as np
from numpy import around
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
#  Metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Models
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# Import Data
df = pd.read_csv("../input/telecommunications-churn/train-scaled.csv")
# Change Churn from string (yes/no) to binary.
y = pd.Series(np.where(df.Churn.values == 'Yes', 1, 0),
          df.index)
# Gather the ID
customer_id = df["customerID"]
# Training data
X = df.drop(["Unnamed: 0", "customerID", "Churn"], axis=1)
X.shape
# Feature Selection algorithm
from pymrmr import mRMR
# Selects the import features
f_select = mRMR(X, 'MID', 10)
# Important features
f_select
from sklearn.decomposition import PCA
pca = PCA()  
XX = pca.fit_transform(X[f_select]) # XX is out new training matrix that has two basis vectors
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size = 0.3, random_state = 0)
# prepare configuration for cross validation test harness
seed = 0
# prepare models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=10, criterion='gini', random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', probability=True, random_state=seed)))
models.append(('ADA', AdaBoostClassifier(n_estimators=50, learning_rate=1,random_state=seed)))
models.append(('XGB', XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)))


# evaluate each model in turn
cross_val_results = list()
accuracy_results = list()
names = list()
scoring = 'accuracy'

for name, model in models:
    print("-" * 70)
    print("This is {name} model.".format(name=name))
    print("-" * 70)
    # Test each model on one run through    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Model statistics for using the test data in the training set
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results.append(accuracy)
    print('This is the accuracy score: {}'.format(accuracy))
    print('Here is the classification report: ')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix')
    print(cm)
    print('\n')
    
    # Test model on its cross-validation score
    cv_results = cross_val_score(model, X_test, y_test, cv=10, scoring=scoring)
    cross_val_results.append(cv_results)
    names.append(name)
    msg = "Cross validation score of %s Model: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print("-" * 70)
    print('\n\n\n')