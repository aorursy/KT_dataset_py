import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics 
df = pd.read_csv('../input/diabetes.csv')
df.head()
X = df.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(roc_auc))
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
predictions = cross_val_predict(model, X_test, y_test, cv=5)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score, decimals=4)
regression = LogisticRegression(C=0.7,random_state = 60)
regression.fit(X_train,y_train)
predicts = regression.predict(X_test)
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, predict, labels=[1, 0]))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, predict, labels=[1, 0]))
scores_train = np.mean(cross_val_score(regression, X_train, y_train, cv=5, scoring='roc_auc'))
np.around(scores_train, decimals=4)
scores_test = np.mean(cross_val_score(regression, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(scores_test, decimals=4)
