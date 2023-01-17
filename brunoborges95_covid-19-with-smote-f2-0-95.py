import warnings

warnings.filterwarnings("ignore") 
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import eli5

from eli5.sklearn import PermutationImportance

import missingno as msno 

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

import sklearn.metrics as metrics

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

base = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

msno.bar(base) 
columns = ['Patient age quantile', 'SARS-Cov-2 exam result','Patient addmited to regular ward (1=yes, 0=no)',

           'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)',

           'Hematocrit', 'Hemoglobin', 'Platelets', 'Mean platelet volume ', 'Red blood Cells', 'Lymphocytes', 'Mean corpuscular hemoglobin concentration (MCHC)',

           'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)', 'Eosinophils', 'Mean corpuscular volume (MCV)',

           'Monocytes', 'Red blood cell distribution width (RDW)', 'Serum Glucose', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 

           'CoronavirusNL63', 'Rhinovirus/Enterovirus','Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43' ,

           'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2' ]



base = base[columns]
base_new = base.dropna(subset=['Hematocrit'])

base_new = base_new.fillna(-9)
from sklearn.preprocessing import LabelEncoder

for col in base_new.columns:

    if base_new[col].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(base_new[col].values))

        base_new[col] = lbl.transform(list(base_new[col].values))
msno.bar(base_new)
matrix = np.triu(base_new.corr())

plt.figure(figsize=(15,15))

sns.heatmap(base_new.corr(), mask=matrix)
print('Negative:', round(

        base_new['SARS-Cov-2 exam result'].value_counts()[0]/len(base_new)*100, 2), '% of the dataset')

print('Positive:', round(

        base_new['SARS-Cov-2 exam result'].value_counts()[1]/len(base_new)*100, 2), '% of the dataset')

sns.countplot('SARS-Cov-2 exam result',data=base_new)
#oversampling of the data. The number of Fraud was twiced

from imblearn.over_sampling import SMOTE

k = 6

sm = SMOTE(k_neighbors=5, random_state=12, n_jobs=8, sampling_strategy={1:int(83*k), 0:int(520)})





#A proporção de classes da variável pre_approved na base de treino e de teste deve ser a mesma da base original

X = base_new.drop(['SARS-Cov-2 exam result'], axis=1)

y = base_new['SARS-Cov-2 exam result']

X, y = sm.fit_resample(X, y)

base_balanced = pd.concat([X, y], axis=1)

sns.countplot('SARS-Cov-2 exam result',data=base_balanced)
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

sss.get_n_splits(X, y)

for train_index, test_index in sss.split(X, y):

    X_train, X_test = X.values[train_index], X.values[test_index]

    y_train, y_test = y.values[train_index], y.values[test_index]



weight = base_new['SARS-Cov-2 exam result'].value_counts()[0]/(6*base_new['SARS-Cov-2 exam result'].value_counts()[1])  

xgboost = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,

       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,

       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,

       objective='binary:logistic', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=weight, seed=0, silent=True, subsample=1)
xgb_calibrated = CalibratedClassifierCV(xgboost, cv=5, method='isotonic')

xgb_calibrated.fit(X_train, y_train)

y_hat = xgb_calibrated.predict_proba(X_test)[:,1]
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_test, y_hat)

roc_auc = metrics.auc(fpr, tpr)

print('The area under the curve ROC is %0.4f:' %roc_auc)



plt.title('Receiver Operating Characteristic (curva ROC)')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
scores_xgb=[] 

tresholds = np.linspace(0 , 1 , 200)

for treshold in tresholds:

    y_hat_xgb = (y_hat > treshold).astype(int)

    scores_xgb.append([metrics.recall_score(y_pred=y_hat_xgb, y_true=y_test),

                 metrics.precision_score(y_pred=y_hat_xgb, y_true=y_test),

                 metrics.fbeta_score(y_pred=y_hat_xgb, y_true=y_test, beta=2),

                 metrics.accuracy_score(y_pred=y_hat_xgb, y_true=y_test)])

scores_xgb = np.array(scores_xgb)

final_tresh = tresholds[scores_xgb[:, 3].argmax()]

y_hat_xgb = (y_hat > final_tresh).astype(int)

best_score = scores_xgb[scores_xgb[:, 3].argmax(),:]

recall_score = best_score[0]

precision_score = best_score[1]

fbeta_score = best_score[2]

acuraccy = best_score[3]



print('The recall score is: %.3f' % recall_score)

print('The precision score is: %.3f' % precision_score)

print('The f2 score is: %.3f' % fbeta_score)

print('The accuracy score is: %.3f' % acuraccy)
cm = pd.crosstab(y_test, y_hat_xgb, rownames=['Real'], colnames=['Predict'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Negative', 'Positive'],

            yticklabels=['Negative', 'Positive'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues", fmt="d")

plt.title('Confusion Matrix for XGBoost', fontsize=14)

plt.show()
perm = PermutationImportance(xgb_calibrated).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = base_new.drop(['SARS-Cov-2 exam result'], axis=1).columns.tolist())