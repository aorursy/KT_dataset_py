!pip install catboost

!pip install ipywidgets

!jupyter nbextension enable --py widgetsnbextension



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier, Pool, cv

from xgboost.sklearn import XGBClassifier

import xgboost as xgb

from tqdm import tqdm



from sklearn.metrics import f1_score, log_loss, confusion_matrix,classification_report

import scikitplot as skplt

from sklearn.metrics import accuracy_score, roc_curve, auc

from sklearn.preprocessing import StandardScaler
data_train = pd.read_csv('../input/aps_failure_training_set.csv')

data_test = pd.read_csv('../input/aps_failure_test_set.csv')
data_train.head()
data_train.isnull().sum()
data_test.isnull().sum()
# NA replacemenet

data_train.replace('na','-1', inplace=True)

data_test.replace('na','-1', inplace=True)
#categorical encoding

data_train['class'] = pd.Categorical(data_train['class']).codes

data_test['class'] = pd.Categorical(data_test['class']).codes



print(['neg', 'pos'])

print(np.bincount(data_train['class'].values))

print(np.bincount(data_test['class'].values))
import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

sns.countplot(x='class', data=data_train, palette='hls')

plt.show()
# split train and test data into X_train,X_test and y_train,y_test

y_train = data_train['class'].copy(deep=True)

X_train = data_train.copy(deep=True)

X_train.drop(['class'], inplace=True, axis=1)



y_test = data_test['class'].copy(deep=True)

X_test = data_test.copy(deep=True)

X_test.drop(['class'], inplace=True, axis=1)



# strings to float

X_train = X_train.astype('float64')

X_test = X_test.astype('float64')
cat_features = list(range(0, X_train.shape[1]))

print(cat_features)
print(X_train.dtypes)

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
def evaluate(y_test,y_pred,y_pred_proba):

    if len(y_pred)>0:

        f1 = f1_score(y_test,y_pred,average="weighted")

        print("F1 score: ",f1)

    if len(y_pred_proba)>0:

        logloss = log_loss(y_test,y_pred_proba, eps=1e-15, normalize=True, sample_weight=None, labels=None)

        print("Log loss for predicted probabilities:",logloss)
forest_clf = RandomForestClassifier(n_estimators=250,n_jobs=-1)

forest_clf.fit(X_train,y_train)

y_pred_rf = forest_clf.predict(X_test)

y_pred_proba_rf = forest_clf.predict_proba(X_test)

evaluate(y_test,y_pred_rf,y_pred_proba_rf)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf ).ravel()

skplt.metrics.plot_confusion_matrix(y_test, y_pred_rf, normalize=False)

plt.show()

print(classification_report(y_test,y_pred_rf))
#display ROC curve

from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)

roc_auc = auc(fpr, tpr)





plt.figure()

plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
y_test_predictions_rec = y_pred_proba_rf[:,1] > 0.1

y_test_predictions_prec = y_pred_proba_rf[:,1] > 0.85
skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_prec, normalize=False)

plt.show()

print(classification_report(y_test, y_test_predictions_prec))
roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
scores = forest_clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
min_cost = np.inf

best_threshold = 0.5

costs = []

for threshold in tqdm(thresholds):

    y_pred_threshold = scores > threshold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    cost = 10*fp + 500*fn

    costs.append(cost)

    if cost < min_cost:

        min_cost = cost

        best_threshold = threshold

print("Best threshold: {:.4f}".format(best_threshold))

print("Min cost: {:.2f}".format(min_cost))
y_pred_test_rf = forest_clf.predict_proba(X_test)[:,1] > best_threshold

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_rf).ravel()

skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_rf, normalize=False)

10*fp + 500*fn
bayes_clf = GaussianNB()

bayes_clf.fit(X_train,y_train)
y_pred__bayes = bayes_clf.predict(X_test)

y_pred_proba_bayes = bayes_clf.predict_proba(X_test)
evaluate(y_test,y_pred__bayes,y_pred_proba_bayes)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred__bayes).ravel()

skplt.metrics.plot_confusion_matrix(y_test,y_pred__bayes, normalize=False)

plt.show()

print(classification_report(y_test,y_pred__bayes))
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test,y_pred__bayes)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
xgb_clf = XGBClassifier(max_depth=5)

xgb_clf.fit(X_train,y_train)
y_pred_xgb = xgb_clf.predict(X_test)

y_pred_proba_xgb = xgb_clf.predict_proba(X_test)
evaluate(y_test,y_pred_xgb,y_pred_proba_xgb)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_xgb).ravel()

skplt.metrics.plot_confusion_matrix(y_test,y_pred_xgb, normalize=False)

plt.show()

print(classification_report(y_test,y_pred_xgb))
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test,y_pred_xgb)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
y_test_predictions_rec = y_pred_proba_xgb[:,1] > 0.1

y_test_predictions_prec = y_pred_proba_xgb[:,1] > 0.85
skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_prec, normalize=False)

plt.show()

print(classification_report(y_test, y_test_predictions_prec))
scores = xgb_clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
min_cost = np.inf

best_threshold = 0.5

costs = []

for threshold in tqdm(thresholds):

    y_pred_threshold = scores > threshold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    cost = 10*fp + 500*fn

    costs.append(cost)

    if cost < min_cost:

        min_cost = cost

        best_threshold = threshold

print("Best threshold: {:.4f}".format(best_threshold))

print("Min cost: {:.2f}".format(min_cost))
y_pred_test_xgb = xgb_clf.predict_proba(X_test)[:,1] > best_threshold

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_xgb).ravel()

skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_xgb, normalize=False)

10*fp + 500*fn
y_test.shape
X_test.shape
X_train.fillna(-999, inplace=True)

X_test.fillna(-999, inplace=True)
model = CatBoostClassifier(

    custom_loss=['Accuracy'],

    random_seed=42,

    logging_level='Silent'

)

model.fit(

    X_train, y_train,

    cat_features=categorical_features_indices,

    eval_set=(X_test, y_test),

    logging_level='Verbose',  # you can uncomment this for text output

    plot=True

);
model = CatBoostClassifier(

    iterations=450,

    random_seed=38,

    learning_rate=0.2,

    eval_metric="Accuracy",

    use_best_model=False

)



model.fit(

    X_train, y_train,

    cat_features=categorical_features_indices,

    eval_set=(X_test, y_test),

    verbose=False,

    plot=True

)
params = {}

params['loss_function'] = 'Logloss'

params['iterations'] = 450

params['custom_loss'] = 'AUC'

params['random_seed'] = 60

params['learning_rate'] = 0.2



cv_data = cv(

    params = params,

    pool = Pool(X_train, label=y_train, cat_features=categorical_features_indices),

    fold_count=5,

    inverted=False,

    shuffle=True,

    partition_random_seed=0,

    plot=True,

    stratified=True,

    verbose=False

)
print(model.predict_proba(data=X_test))
import scikitplot as skplt
y_pred = model.predict(data=X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)

y_pred_proba = model.predict_proba(X_test)

plt.show()

print(classification_report(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
y_test_predictions_high_precision = y_pred_proba[:,1] > 0.8

y_test_predictions_high_recall = y_pred_proba[:,1] > 0.1
skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_high_precision, normalize=False)

plt.show()

print(classification_report(y_test, y_test_predictions_high_precision))
10*120+ 500*9
scores = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)



plt.figure()

plt.plot(fpr, tpr, color='red',label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
min_cost = np.inf

best_threshold = 0.5

costs = []

for threshold in tqdm(thresholds):

    y_pred_threshold = scores > threshold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    cost = 10*fp + 500*fn

    costs.append(cost)

    if cost < min_cost:

        min_cost = cost

        best_threshold = threshold

print("Best threshold: {:.4f}".format(best_threshold))

print("Min cost: {:.2f}".format(min_cost))
y_pred_test_final = model.predict_proba(X_test)[:,1] > best_threshold

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test_final).ravel()

skplt.metrics.plot_confusion_matrix(y_test,y_pred_test_final, normalize=False)

10*fp + 500*fn