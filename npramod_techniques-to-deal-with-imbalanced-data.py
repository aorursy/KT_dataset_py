import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline



from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve

from imblearn.over_sampling import SMOTE, RandomOverSampler

from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler

from imblearn.combine import SMOTEENN,SMOTETomek

from imblearn.ensemble import BalanceCascade

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../input/creditcard.csv")

data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

print(count_classes)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))

data = data.drop(['Time'],axis=1)

data.head()
X = data.iloc[:, data.columns != 'Class']

y = data.iloc[:, data.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print(X_train.shape)

print(X_test.shape)
def benchmark(sampling_type,X,y):

    lr = LogisticRegression(penalty = 'l1')

    param_grid = {'C':[0.01,0.1,1,10]}

    gs = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2)

    gs = gs.fit(X.values,y.values.ravel())

    return sampling_type,gs.best_score_,gs.best_params_['C']



def transform(transformer,X,y):

    print("Transforming {}".format(transformer.__class__.__name__))

    X_resampled,y_resampled = transformer.fit_sample(X.values,y.values.ravel())

    return transformer.__class__.__name__,pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)
datasets = []

datasets.append(("base",X_train,y_train))

datasets.append(transform(SMOTE(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomOverSampler(),X_train,y_train))

#datasets.append(transform(ClusterCentroids(n_jobs=-1),X_train,y_train))

datasets.append(transform(NearMiss(n_jobs=-1),X_train,y_train))

datasets.append(transform(RandomUnderSampler(),X_train,y_train))

datasets.append(transform(SMOTEENN(),X_train,y_train))

datasets.append(transform(SMOTETomek(),X_train,y_train))







benchmark_scores = []

for sample_type,X,y in datasets:

    print('______________________________________________________________')

    print('{}'.format(sample_type))

    benchmark_scores.append(benchmark(sample_type,X,y))

    print('______________________________________________________________')

    
benchmark_scores
scores = []

# train models based on benchmark params

for sampling_type,score,param in benchmark_scores:

    print("Training on {}".format(sampling_type))

    lr = LogisticRegression(penalty = 'l1',C=param)

    for s_type,X,y in datasets:

        if s_type == sampling_type:

            lr.fit(X.values,y.values.ravel())

            pred_test = lr.predict(X_test.values)

            pred_test_probs = lr.predict_proba(X_test.values)

            probs = lr.decision_function(X_test.values)

            fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),pred_test)

            p,r,t = precision_recall_curve(y_test.values.ravel(),probs)

            scores.append((sampling_type,

                           f1_score(y_test.values.ravel(),pred_test),

                           precision_score(y_test.values.ravel(),pred_test),

                           recall_score(y_test.values.ravel(),pred_test),

                           accuracy_score(y_test.values.ravel(),pred_test),

                           auc(fpr, tpr),

                           auc(p,r,reorder=True),

                           confusion_matrix(y_test.values.ravel(),pred_test)))



sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])

sampling_results
lr = LogisticRegression(penalty = 'l1',class_weight="balanced")

lr.fit(X_train.values,y_train.values.ravel())

scores = []

pred_test = lr.predict(X_test.values)

pred_test_probs = lr.predict_proba(X_test.values)

probs = lr.decision_function(X_test.values)

fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),pred_test)

p,r,t = precision_recall_curve(y_test.values.ravel(),probs)

scores.append(("weighted_base",

                           f1_score(y_test.values.ravel(),pred_test),

                           precision_score(y_test.values.ravel(),pred_test),

                           recall_score(y_test.values.ravel(),pred_test),

                           accuracy_score(y_test.values.ravel(),pred_test),

                           auc(fpr, tpr),

                           auc(p,r,reorder=True),

                           confusion_matrix(y_test.values.ravel(),pred_test)))



scores = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])
results = sampling_results.append(scores)

results