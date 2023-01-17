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
import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning) # Предупреждение о достижении максимального количества итераций
warnings.filterwarnings(action='ignore', category=FitFailedWarning)   # Предупреждение о не выполненном обучении

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/data.xlsx")
df["target"].value_counts() 
Y = df["target"] 
X = df.drop("target", axis=1, inplace=False)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X) # стандартизация признаков для равноправного их влияния на функцию потерь регрессии
X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.33, random_state=42) # разделяем выборку на обучающуюся и тестовую
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
clf.fit(X_train, y_train) # обучение классификатора
y_pred = clf.decision_function(X_test) # результаты регрессии для каждого события в тесте
precision, recall, thresholds = precision_recall_curve(y_test, y_pred) # thresholds - порог разделения классов
plt.plot(thresholds, precision[:-1]) # точность при разных порогах
plt.plot(thresholds, recall[:-1]) # полнота при разных порогах (чем больше, тем меньше событий попадает в класс '1')
plt.plot(recall, precision)
average_precision_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
roc_auc_score(y_test, y_pred)
#Можно сделать вывод, что модель не точная, так как график имеет слабую выпуклость и маленькую площадь под графиком
parameters_grid = {
    'penalty': ['l1', 'l2', 'none'], # Методы регуляризации
    'C': np.append(np.linspace(0.05, 2.5, num=5), np.nan), # параметр регуляризации
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # алгоритмы оптимизации
    'max_iter': np.linspace(20, 300, num=5) # Количество итераций метода
}

grid_cv = GridSearchCV(clf, parameters_grid, scoring = 'accuracy', cv = 10)
grid_cv.fit(X, Y)
grid_cv.best_params_
grid_cv.best_score_
grid_cv.best_estimator_