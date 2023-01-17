import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train.head()
train.info()
print(train.isnull().sum())

print(test.isnull().sum())
train.describe()
train.groupby('SmokingStatus').count()
train.groupby('Sex').count()
train['Sex'] = train['Sex'].map({'Male':0,'Female':1})
test['Sex'] = test['Sex'].map({'Male':0,'Female':1})
train.head()
train = pd.get_dummies(train,columns=['SmokingStatus'],drop_first=True)
test = pd.get_dummies(test,columns=['SmokingStatus'],drop_first=True)
train.head()
sns.pairplot(train)
sns.heatmap(train.corr(),cmap = 'RdYlBu_r')
cat_cols =[col for col in train.columns if train[col].dtype==object]
cat_cols
from sklearn.preprocessing import LabelEncoder

cat_encs = []

for col in cat_cols:

    le = LabelEncoder()

    train[col] = le.fit_transform(train[col])

    cat_encs.append([col,le])
cat_encs
cat_test = []

for col in cat_cols:

    le = LabelEncoder()

    test[col] = le.fit_transform(test[col])

    cat_test.append([col,le])
cat_test
X =train.drop('Patient',axis=1)

y= train['Patient']
X.shape
y.shape
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X,y)

y_predict=model.predict(X)

from sklearn.metrics import accuracy_score

accuracy_score(y,y_predict)

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

clf_rf = ensemble.RandomForestClassifier(random_state=1)

parameters = { 

    'n_estimators': [100, 400],

    'criterion' : ['gini', 'entropy'],

    'max_depth' : [2, 4, 6]    

}



from sklearn.model_selection import GridSearchCV, cross_val_score



cv_rf = GridSearchCV(estimator = clf_rf, param_grid = parameters, cv=5, n_jobs=-1)
X_test = test



clf = cv_rf.fit(X, y)



predictions = clf.predict(X_test)
X_test.shape
output = pd.DataFrame({'Patient': test.Patient, 'FVC': predictions})

output.to_csv('submission.csv', index=False)