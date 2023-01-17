import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage 
import xgboost as xg
from imblearn.over_sampling import SMOTE #For imbalanced datasets
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline, Pipeline
train = pd.read_csv('../input/exoTrain.csv')
test =  pd.read_csv('../input/exoTest.csv')
train.head()
X_train = train.drop('LABEL', axis=1)
y_train = train.LABEL
X_test = test.drop('LABEL', axis=1)
y_test = test.LABEL
z=train[train.LABEL==2]
z
for i in [0,1,2,3,4]:
    Y = X_train.iloc[i]
    X = np.arange(len(Y)) 
    plt.figure(figsize=(15,5))
    plt.ylabel('Flux')
    plt.xlabel('Observation')
    plt.plot(X, Y)
    plt.show()


for i in [j for j in range (37,42)]:    
    Y = X_train.iloc[i]
    X = np.arange(len(Y)) 
    plt.figure(figsize=(15,5))
    plt.ylabel('Flux')
    plt.xlabel('Observation')
    plt.plot(X, Y)
    plt.show()
def normal(X):
    Y= (X-np.mean(X))/(np.max(X)-np.min(X))
    return Y
X_train= X_train.apply(normal,axis=1)

X_test= X_test.apply(normal,axis=1)
def fourier(X):
    Y = scipy.fft(X, n=X.size)
    return np.abs(Y)
X_train = X_train.apply(fourier,axis=1)
X_test = X_test.apply(fourier,axis=1)
for i in [0,1,2,3,4]:
    Y = X_train.iloc[i]
    X = np.arange(len(Y))*(1/(36.0*60.0)) 
    plt.figure(figsize=(15,5))
    plt.ylabel('Flux')
    plt.xlabel('Frequency')
    plt.plot(X, Y)
    plt.show()
for i in [j for j in range (37,42)]:
    Y = X_train.iloc[i]
    X = np.arange(len(Y))*(1/(36.0*60.0)) 
    plt.figure(figsize=(15,5))
    plt.ylabel('Flux')
    plt.xlabel('Frequency')
    plt.plot(X, Y)
    plt.show()
X_train = X_train.drop(X_train.columns[1601:], axis=1)
X_test = X_test.drop(X_test.columns[1601:], axis=1)
X_train_l = X_train.drop(X_train.columns[1:100], axis=1)
X_test_l = X_test.drop(X_test.columns[1:100], axis=1)
sm = SMOTE(ratio = 'auto')
X_train= X_train.as_matrix()
X_test= X_test.as_matrix()


X_train_l= X_train_l.as_matrix()
X_test_l= X_test_l.as_matrix()



class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            sm = SMOTE(ratio = 'auto')
            model = self.models[key]
            params = self.params[key]
            pipeline = Pipeline([('sm',sm), ('model',model)]) #create a PIPELINE to apply SMOTE
            kf = StratifiedKFold(n_splits=10)
            gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='min_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores)
                 #'std_score': std(scores), Uncomment later
            }
            return pd.Series({**params,**d}) #Fixed for Python 3
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False) #Fixed for Python 3 sort_values
        
        columns = ['estimator', 'min_score', 'max_score']
         #, 'max_score', 'std_score'] uncomment
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]

models1 = { 
    
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'XGBClassifier': xg.XGBClassifier(),

}

params1 = { 
    'RandomForestClassifier': { "model__n_estimators": [10, 18, 22],
              "model__max_depth": [3, 5],
              "model__min_samples_split": [15, 20],
              "model__min_samples_leaf": [5, 10, 20], },
    'AdaBoostClassifier':  { "model__n_estimators": [10, 18, 22],
                            },
              
    'XGBClassifier': { 'model__n_estimators': [1000], 'model__learning_rate': [1.0],'model__max_depth':range(3,10,2),
 'model__min_child_weight':range(1,6,2) },

}
f1_scorer = make_scorer(f1_score, pos_label=2)


helper = EstimatorSelectionHelper(models1, params1)
helper.fit(X_train, y_train, scoring=f1_scorer, n_jobs=-1)

helper.score_summary(sort_by='max_score')
helper = EstimatorSelectionHelper(models1, params1)
helper.fit(X_train_l, y_train, scoring=f1_scorer, n_jobs=-1)

helper.score_summary(sort_by='max_score')
sm = SMOTE(ratio =1.0,random_state=123 )
X_r, y_r = sm.fit_sample(X_train, y_train)
print (y_train.value_counts(), np.bincount(y_r))
model = xg.XGBClassifier(n_estimators = 1000, learning_rate= 1.0, max_depth= 9, min_child_weight = 3,seed=123)

model.fit(X_r, y_r)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
f2_score = f1_score(y_test, predictions,pos_label = 2)
print("f1_score: %.2f%%" % (f2_score * 100.0))
sm = SMOTE(ratio =1.0, random_state=123)
X_l, y_l = sm.fit_sample(X_train_l, y_train)
print (y_train.value_counts(), np.bincount(y_r))
model = xg.XGBClassifier(n_estimators = 1000, learning_rate= 1.0, max_depth= 9, min_child_weight = 3,seed=123)

model.fit(X_l, y_l)
# make predictions for test data
y_pred = model.predict(X_test_l)
predictions = [round(value) for value in y_pred]
# evaluate predictions
f2_score = f1_score(y_test, predictions,pos_label = 2)
print("f1_score: %.2f%%" % (f2_score * 100.0))
model = xg.XGBClassifier(n_estimators = 1000, learning_rate= 1.0, max_depth= 7, min_child_weight = 5,seed=123)

model.fit(X_r, y_r)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
f2_score = f1_score(y_test, predictions,pos_label = 2)
print("f1_score: %.2f%%" % (f2_score * 100.0))
model = xg.XGBClassifier(n_estimators = 1000, learning_rate= 1.0, max_depth= 7, min_child_weight = 3,seed=123)

model.fit(X_l, y_l)
# make predictions for test data
y_pred = model.predict(X_test_l)
predictions = [round(value) for value in y_pred]
# evaluate predictions
f2_score = f1_score(y_test, predictions,pos_label = 2)
print("f1_score: %.2f%%" % (f2_score * 100.0))
y_pred
l = np.random.randint(10000, size=5)
l
for n in l:
    sm = SMOTE(ratio =1.0, random_state= n)
    X_l, y_l = sm.fit_sample(X_train_l, y_train)
#print (y_train.value_counts(), np.bincount(y_r))
    model = xg.XGBClassifier(n_estimators = 1000, learning_rate= 1.0, max_depth= 7, min_child_weight = 3,seed=n)
    model.fit(X_l, y_l)
    # make predictions for test data
    y_pred = model.predict(X_test_l)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    f2_score = f1_score(y_test, predictions,pos_label = 2)
    print("f1_score: %.2f%%" % (f2_score * 100.0))