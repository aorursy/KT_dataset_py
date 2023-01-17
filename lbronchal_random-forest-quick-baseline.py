import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/aps_failure_training_set_processed_8bit.csv')
test_data = pd.read_csv('../input/aps_failure_test_set_processed_8bit.csv')
train_data.head()
train_data['class'].value_counts()
train_data['class'] = train_data['class'].apply(lambda x: 0 if x<=0 else 1)
test_data['class'] = test_data['class'].apply(lambda x: 0 if x<=0 else 1)
X = train_data.drop('class', axis=1)
y = train_data['class']
X_test_final = test_data.drop('class', axis=1)
y_test_final = test_data['class']
desc = X.describe()
desc
X.columns[np.where(desc.loc['std'] < 0.005)].values
corr = X.corr()
np.fill_diagonal(corr.values, 0)
(corr>0.999).sum().sum()/2
corr.abs().unstack().sort_values(kind="quicksort", ascending=False)[:10]
X.isnull().any().any()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, make_scorer
import scikitplot as skplt
from tqdm import tqdm

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
model_full_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)
model_full_rf.fit(X_train, y_train)
model_full_rf.score(X_test, y_test)
model_full_rf.score(X_train, y_train)
y_pred = model_full_rf.predict(X_test)
y_pred_proba = model_full_rf.predict_proba(X_test)
print(classification_report(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()
10*fp + 500*fn
y_test_predictions_high_precision = y_pred_proba[:,1] > 0.8
y_test_predictions_high_recall = y_pred_proba[:,1] > 0.1
skplt.metrics.plot_confusion_matrix(y_test, y_test_predictions_high_recall, normalize=False)
plt.show()
10*232 + 500*17
scores = model_full_rf.predict_proba(X_test)[:,1]
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
plt.figure(figsize=(14,8))
plt.scatter(x=thresholds, marker='.', y=costs)
plt.title('Cost per threshold')
plt.xlabel('threshold')
plt.ylabel('cost')
plt.show()
y_pred_test_final = model_full_rf.predict_proba(X_test_final)[:,1] > best_threshold
tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_test_final).ravel()
10*fp + 500*fn