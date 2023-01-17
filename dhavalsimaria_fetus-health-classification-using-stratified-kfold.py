import numpy as np

from numpy import interp

import pandas as pd

import os

import math
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from statistics import mean

from sklearn import model_selection

from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, classification_report

from sklearn.multiclass import OneVsRestClassifier
fetal_health_df = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")

fetal_health_df.drop_duplicates(inplace = True)
fetal_health_df.head()
fetal_health_df.info()
fetal_health_df.shape
X = fetal_health_df.drop(columns = ['fetal_health'],axis = 1)

y = fetal_health_df['fetal_health'].to_numpy()

y = preprocessing.label_binarize(y, classes=[1.0, 2.0, 3.0])
plt.figure(figsize=(12, 6))

sns.countplot(fetal_health_df['fetal_health'], palette='viridis')

plt.title('Dependent variable distribution plot')

plt.xlabel('Fetal Health')
correlation = fetal_health_df.corr()

plt.figure(figsize=(20, 12))

sns.heatmap(correlation, cmap="coolwarm", annot=True)
col_names = X.columns
feature_selection_classifier = DecisionTreeClassifier()

sfm = SelectFromModel(estimator=feature_selection_classifier)

X_transformed = sfm.fit_transform(X, y)

support = sfm.get_support()
selected_cols = [x for x, y in zip(col_names, support) if y == True]
#X_selected = fetal_health_df[selected_cols]

X_selected=fetal_health_df.loc[:, fetal_health_df.columns.isin(selected_cols)]
X_selected.head()
scaler = preprocessing.StandardScaler() 

X_scaled = scaler.fit_transform(X_selected) 
n_bins = 1 + round(math.log(len(X_selected.axes[0])))

print(n_bins)
stratified_kf = model_selection.StratifiedKFold(n_splits = n_bins, shuffle=True)
fetal_health_classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state = 42))



fig1 = plt.figure(figsize=[12,12])

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

all_f1_score = []

precision_dict = dict()

recall_dict = dict()





for i, (train_index, test_index) in enumerate(stratified_kf.split(X_scaled, y.argmax(1))):

    X_train, X_test = X_scaled[train_index], X_scaled[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    fetal_health_classifier.fit(X_train, y_train)

    y_pred = fetal_health_classifier.predict(X_test)

    prediction_proba = fetal_health_classifier.predict_proba(X_test)

    fpr, tpr, t = roc_curve(y_test[:, 1], prediction_proba[:, 1])

    precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_test[:, 1], prediction_proba[:, 1])

    f1score = f1_score(y_test, y_pred, average='weighted')

    all_f1_score.append(f1score)

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    



mean_fi_score = mean(all_f1_score)

print("Mean F1-score across all folds: ", mean_fi_score)

plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'black')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

print("Mean ROC across all folds: ", mean_auc)

plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC(AUC=%0.2f)' % (mean_auc), lw = 2, alpha=1)

         

         

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(loc="lower right")

plt.show()
fig2 = plt.figure(figsize=[12,12])



for i in range(len(precision_dict)):

    plt.plot(recall_dict[i], precision_dict[i], lw=2, label='Fold %d' % i)

    

    

    

plt.xlabel("recall")

plt.ylabel("precision")

plt.legend(loc="best")

plt.title("precision vs. recall curve")

plt.show()