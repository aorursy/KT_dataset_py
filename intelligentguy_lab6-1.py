# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, average_precision_score, roc_curve, roc_auc_score



from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/data.xlsx")
df.head()
df["target"].value_counts() #классы сбалансированы
Y = df["target"] # отделяем признаки от классов

X = df.drop("target", axis=1, inplace=False)
scaler = StandardScaler()

scaled_X = scaler.fit_transform(X) # стандартизируем признаки для равноправного их влияния на функцию потерь регрессии
X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.33, random_state=42) # разделяем выборку на обучающуюся и тестовую
clf = LogisticRegression(random_state=0).fit(X_train, y_train) # обучаем классификатор (регрессию)

y_pred = clf.decision_function(X_test) # результаты (сумма) регрессии для каждого события в тесте
y_pred
precision, recall, thresholds = precision_recall_curve(y_test, y_pred) # thresholds-порги разделения классов

                                                                       # precision[i], recall[i] соответствуют порогу thresholds[i]
plt.plot(thresholds, precision[:-1]) # точность при разных порогах разделения
plt.plot(thresholds, recall[:-1]) # полнота при разных порогах разделения (Чем больше значение тем меньше событй попадает в класс '1')
plt.plot(recall, precision)

average_precision_score(y_test, y_pred)

#plot_precision_recall_curve(clf, X_test, y_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)

roc_auc_score(y_test, y_pred)
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



# Отключение надоедливых ворингов

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

gs.fit(X, Y)
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