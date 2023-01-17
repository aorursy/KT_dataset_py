import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

%matplotlib inline
fraud_df = pd.read_csv("../input/creditcard.csv")
fraud_df.head()
fraud_df.info()
fraud_df.describe()
print(fraud_df['Class'].value_counts())
sns.countplot(x = 'Class', data = fraud_df)
plt.show()
fraud_df.drop(columns = 'Time', inplace = True)
y = fraud_df.pop('Class').values
X = fraud_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
y_train.sum() / len(y_train)
y_test.sum() / len(y_test)
def get_opt_model(clf, param_grid, X_train, y_train, scoring = 'accuracy'):
    '''
    PARAMETERS:
    clf(classifier): classifier to run GridSearch on
    param_grid(dict): dictionary of parameters to test in GridSearch
    X_train(array): array of predictor variables
    y_train(array): array of response variable
    scoring: score to maximize during GridSearch, default to accuracy
    PRINTS: best training score and best parameter values determined by GridSearch
    RETURNS: fitted model (based on entire train set) with best parameter values
    '''

    best_clf = GridSearchCV(clf, param_grid, scoring = scoring, cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    print(best_clf.best_score_)
    print(best_clf.best_params_)
    return best_clf.best_estimator_
smt = SMOTE()
log_model = LogisticRegression()

steps = [('smote', smt), ('mod', log_model)]
pipeline = Pipeline(steps)

c_param_grid = {'mod__C': [0.01, 0.1, 1, 10, 100], 'mod__penalty': ['l1', 'l2']}
best_mod = get_opt_model(pipeline, c_param_grid, X_train_sc, y_train, scoring = 'recall')
fraud_pred_train = best_mod.predict(X_train_sc)
fraud_pred_test = best_mod.predict(X_test_sc)
recall_train = recall_score(y_train, fraud_pred_train)
recall_test = recall_score(y_test, fraud_pred_test)
print('Recall - Training set: ', recall_train)
print('Recall - Test set: ', recall_test)
print(confusion_matrix(y_test, fraud_pred_test))
pos_probs = best_mod.predict_proba(X_test_sc)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in thresholds: 
    fraud_pred = pos_probs > t
    confusion = confusion_matrix(y_test, fraud_pred)
    print('Recall for threshold, ', t, ' is:', confusion[1,1]/(confusion[1,0] + confusion[1,1]))
    print('Precision for threshold, ', t, ' is:', confusion[1,1]/(confusion[0,1] + confusion[1,1]))
    print('-----------------------')
