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
df=pd.read_excel("/kaggle/input/cred.xlsx")
df = df.drop("Дата кредитования", axis = 1)
df.head()
from sklearn.tree import DecisionTreeClassifier
dfpp = df.drop(["Сумма кредита","Стоимость кредита","Срок кредита","Количество","Возраст","Площадь квартиры","Срок эксплуатации машины","Время работы предприятия","Срок работы на предприятии","Срок работы по специальности","Среднемес. доход","Среднемес. расход","Количество иждевенцев","Срок проживания в регионе"], axis =1)
dfpp_a = np.array(dfpp)
test = []
for i in dfpp:
    test.append(pd.unique(df[i]).tolist()) 
test_a = np.array(test)
dfpp_a
dfpp_aa = np.zeros((dfpp.shape[0],dfpp.shape[1]))
for i in range(dfpp.shape[1]):
    for j in range(dfpp.shape[0]):
        for ii in range(len(test_a[i])):
            if dfpp_a[j][i]==test_a[i][ii]:
                dfpp_aa[j][i]=(int)(ii)
dfpp_aa
test2 = list(dfpp)
ii=0
dfpp_aa=dfpp_aa.transpose()
print(dfpp_aa.shape)
for i in test2:
    df.drop(i, axis =1)
    df[i] = dfpp_aa[ii].transpose()
    ii+=1
df
from sklearn.model_selection import train_test_split
X = df.drop("Давать кредит", axis = 1)
y = df[["Давать кредит"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf,feature_names = X.columns, class_names=["Нет", "Да"],filled = True) 
clf.score(X_test, y_test)
result = pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
result = result[result["importance"]>0.05].sort_values("importance", ascending=False)
result
from sklearn.model_selection import train_test_split, GridSearchCV
parameters = dict(max_depth=np.arange(3, 10),
                  min_samples_split=np.arange(3, 15),
                  min_samples_leaf=np.arange(1, 6))
clf = DecisionTreeClassifier(random_state=1)
gs = GridSearchCV(clf, parameters, scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=7, return_train_score=True, refit="accuracy")
gs.fit(X, y)
results = pd.concat(
    [pd.DataFrame(gs.cv_results_["params"])] + 
        [
            pd.DataFrame(gs.cv_results_["mean_test_" + metric], columns=[metric])
                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        ] + 
        [
            pd.DataFrame(gs.cv_results_["mean_train_" + metric], columns=["train " + metric])
                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        ],
axis=1)
results
gs.best_params_, gs.best_score_
dupl = results.duplicated(subset=["accuracy", "f1", "precision", "recall", "roc_auc",
                                  "train accuracy", "train f1", "train precision", "train recall", "train roc_auc"], keep="first")
results = results[~dupl]
results.sort_values("accuracy", ascending=False).head(10)
X = X[['Сумма кредита', 'Расположение', 'Срок кредита', 'Среднемес. доход','Среднемес. расход']]
gs.fit(X, y)
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
        [pd.DataFrame(gs.cv_results_['std_test_accuracy'], columns=["std_test_accuracy"])],
    axis=1)
dupl = results.duplicated(subset=["accuracy", "f1", "precision", "recall", "roc_auc", "train accuracy",], keep="first")
results = results[~dupl]
results.sort_values("accuracy", ascending=False).head(10)

result = pd.DataFrame({"feature": X.columns, "importance": gs.best_estimator_.feature_importances_})
result = result[result["importance"]>0.05].sort_values("importance", ascending=False)
result
tree.plot_tree(gs.best_estimator_,
               feature_names = X.columns,
               class_names=["Нет", "Да"],
               filled = True)