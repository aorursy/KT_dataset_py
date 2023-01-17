import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.SeniorCitizen = data.SeniorCitizen.map(lambda x: 'Yes' if x == 1 else 'No') # Convert values 1 and 0 to "Yes" and "No"
data['TotalCharges'] = data['TotalCharges'].replace(" ", "")                     # Delete the whitespaces 
data.TotalCharges = pd.to_numeric(data.TotalCharges)                             # Convert to float type
X = data.drop(['customerID','Churn'], axis=1)                                    # Drop dependent and ID features
X.shape
X.head(3)
categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
numerical_columns   = [c for c in X.columns if X[c].dtype.name != 'object']
print('List of categorical columns: {:}.\n \nList of numerical: {:}'.format(categorical_columns, numerical_columns))

X[categorical_columns].describe()
X[numerical_columns].describe()
X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].describe()['mean'])
data.corr() # Check linear dependence
def plotfeatures(col1, col2):

    plt.figure(figsize=(10, 6))

    plt.scatter(X[col1][data['Churn'] == 'Yes'],
                X[col2][data['Churn'] == 'Yes'],
                alpha=0.75,
                color='red',
                label='Yes')

    plt.scatter(X[col1][data['Churn'] == 'No'],
                X[col2][data['Churn'] == 'No'],
                alpha=0.75,
                color='blue',
                label='No')

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend(loc='best');
plotfeatures('tenure', 'TotalCharges')
plotfeatures('MonthlyCharges', 'TotalCharges')
data['MonthlyCharges_new'] = data['TotalCharges']/data['tenure']
data.corr()
data.columns
X = X.drop(['tenure','TotalCharges'], axis=1)
binary_columns    = [c for c in categorical_columns if X[str(c)].describe()['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if X[str(c)].describe()['unique'] > 2]
nonbinary_columns
binary_columns
for c in binary_columns:
    top = X[str(c)].describe()['top']
    top_items = X[c] == top
    X.loc[top_items, c] = 0
    X.loc[np.logical_not(top_items), c] = 1
    
X_dummy = pd.get_dummies(X[nonbinary_columns])
X = X.drop(nonbinary_columns, axis=1)
X['MonthlyCharges'] = (X['MonthlyCharges'] - X['MonthlyCharges'].mean()) / X['MonthlyCharges'].std()
X_full = pd.concat((X, X_dummy), axis=1)

data.at[data['Churn'] == 'No', 'Churn'] = 0
data.at[data['Churn'] == 'Yes', 'Churn'] = 1
y = data.Churn
X_full.head(4)
cv = KFold(n_splits=5, shuffle=True) 

scoring = [ 'f1', 'precision', 'recall', 'roc_auc']


for score in scoring:
    lr = linear_model.LogisticRegression()
    scores = np.mean(cross_val_score(lr, X_full, y,
                                 scoring=score,
                                 cv=cv))

    print('{} score: {}'.format(score, scores))

from sklearn import ensemble
# RandomForest can give important features. Then, this important features can use for a new model

scoring = [ 'f1', 'precision', 'recall', 'roc_auc']
for score in scoring:
    
    rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
    scores = np.mean(cross_val_score(rf, X_full, y,
                             scoring=score,
                             cv=cv))
    
    print('{} score: {}'.format(score, scores))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size = 0.3, random_state = 11)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

d_first = 30
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(X_full.columns)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);
best_features = indices[:15]
best_features_names = X_full.columns[best_features]
print(best_features_names)
gbt = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train[best_features_names], y_train)

quality = np.mean(y_test == gbt.predict(X_test[best_features_names]))
print(quality)
