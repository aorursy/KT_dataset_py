# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/data.xlsx")
list1 = df["target"]
df.head()
df = df.drop("target", axis = 1)
df.head()
scaler = StandardScaler()
scaler.fit(df)
scaler.mean_
dfst = pd.DataFrame(scaler.transform(df))
dfst.head()
X_train, X_test, y_train, y_test = train_test_split(df, list1, test_size=0.2, random_state=0)
logistic_regression = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred_2=logistic_regression.predict_proba(X_test)
y_pred=y_pred_2[:,1]
from sklearn.metrics import precision_recall_curve
y_true = y_test
y_scores = y_pred
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
print(precision)
print(recall)
print(thresholds)
plt.plot(recall,precision)
plt.show()
from sklearn.metrics import average_precision_score
average_precision_score(y_true, y_scores)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_scores)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
#print(fpr)
#print(tpr)
#print(thresholds)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.model_selection import GridSearchCV
class RaisingLogisticRegression(LogisticRegression):   
    def fit(self, *args, **kwargs):
        """
        Так как при penalty="none" параметр C игнорируется, то нет смысла его перебирать и вычислять одно и то же.
        """
        if (not np.isnan(self.C)) == (self.penalty == "none"):
            raise ValueError(f"Not allowed!!! C={self.C}, penalty={self.penalty}")
        if self.penalty == "none":
            self.C = 1.0
        return super().fit(*args, **kwargs)
    
clf = RaisingLogisticRegression(random_state=0)

import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning) # Предупреждение о достижении максимального количества итераций
warnings.filterwarnings(action='ignore', category=FitFailedWarning)   # Предупреждение о не выполненном обучении

parameters = {
    'penalty': ['l1', 'l2', 'none'], # Методы регуляризации
    'C': np.append(np.linspace(0.05, 2.5, num=5), np.nan), # параметр регуляризации
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # алгоритмы оптимизации
    'max_iter': np.linspace(20, 300, num=5) # Количество итераций метода
}
gs = GridSearchCV(clf, parameters, scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=10, return_train_score=True, refit="accuracy")
gs.fit(df, list1)
list(gs.cv_results_.keys())
results = pd.concat(
    [pd.DataFrame(gs.cv_results_["params"])] + 
        [
            pd.DataFrame(gs.cv_results_["mean_test_" + metric], columns=[metric])
                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        ] + 
        [
            pd.DataFrame(gs.cv_results_["mean_train_" + metric], columns=["train " + metric])
                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        ] + 
        [pd.DataFrame(gs.cv_results_["mean_fit_time"], columns=["mean_fit_time"])],
    axis=1)
results
results = results[~results.loc[:, 'accuracy':'roc_auc'].isnull().all(axis=1)]
dupl = results.duplicated(subset=["C",	"penalty", "solver", "accuracy", "f1", "precision", "recall", "roc_auc",
                                  "train accuracy", "train f1", "train precision", "train recall", "train roc_auc"], keep="first")
results = results[~dupl]
results
results.sort_values("accuracy", ascending=False).head(10)
results.sort_values("f1", ascending=False).head(10)
results.sort_values("roc_auc", ascending=False).head(10)
gs.best_params_
gs.best_score_
gs.best_estimator_