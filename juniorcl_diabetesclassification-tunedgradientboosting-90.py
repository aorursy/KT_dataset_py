# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
base = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
np.random.seed(42)
base.head()
sns.barplot(base.Outcome, base.Outcome.value_counts());
plt.grid()
base.describe().T
sns.pairplot(base, hue="Outcome", vars=base.columns[1:-1])
plt.show()
na_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
base[na_columns] = base[na_columns].replace(0, np.nan)
base.isna().mean() * 100
df = base.copy()

for column in ["Glucose", "SkinThickness", "Insulin"]:
    median_0 = base[column][base["Outcome"] == 0].median()
    median_1 = base[column][base["Outcome"] == 1].median()
    
    df[column][df["Outcome"] == 0] = base[column][df["Outcome"] == 0].fillna(median_0)
    df[column][df["Outcome"] == 1] = base[column][df["Outcome"] == 1].fillna(median_1)
df.BloodPressure.fillna(df.BloodPressure.median(), inplace=True)
df.BMI.fillna(df.BMI.median(), inplace=True)
X = df.drop("Outcome", axis=1)
X.head()
y = df.Outcome
y.head()
f1 = metrics.make_scorer(metrics.f1_score)
accuracy = metrics.make_scorer(metrics.accuracy_score)
precision = metrics.make_scorer(metrics.precision_score)
recall = metrics.make_scorer(metrics.recall_score)
auc = metrics.make_scorer(metrics.roc_auc_score)
scoring = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
def printResults(cv):
    print("Accuracy  {:.3f} ({:.3f})".format(cv["test_accuracy"].mean(), cv["test_accuracy"].std()))
    print("Precision {:.3f} ({:.3f})".format(cv["test_precision"].mean(), cv["test_precision"].std()))
    print("Recall    {:.3f} ({:.3f})".format(cv["test_recall"].mean(), cv["test_recall"].std()))
    print("F1        {:.3f} ({:.3f})".format(cv["test_f1"].mean(), cv["test_f1"].std()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues");
cv_gbc = cross_validate(gbc, X, y, scoring=scoring, cv=5)
printResults(cv_gbc)
params = {
    "loss": ["deviance", "exponential"],
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

gs = GridSearchCV(estimator=gbc, param_grid=params, cv=5)
gs.fit(X, y)
gs.best_score_
gs.best_params_
gbc_best = GradientBoostingClassifier(learning_rate=0.2, loss='exponential', n_estimators=100)
gbc_best.fit(X_train, y_train)
y_pred = gbc_best.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues");
cv_gbc_best = cross_validate(gbc_best, X, y, cv=5, scoring=scoring)
printResults(cv_gbc_best)
