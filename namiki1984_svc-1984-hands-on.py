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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from pprint import pprint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, roc_curve, auc,roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from timeit import time
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
train["Age"]=train["Age"].fillna(train["Age"].median())
train["Embarked"]=train["Embarked"].fillna("S")

test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Embarked"]=test["Embarked"].fillna("S")

train = pd.get_dummies(train, columns=["Sex","Pclass","Embarked"])
test = pd.get_dummies(test, columns=["Sex","Pclass","Embarked"])

train["FamilyNum"] = train["SibSp"] + train["Parch"]
train["hasFamily"] = train["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train = train.drop(labels = ["SibSp"], axis = 1)
train = train.drop(labels = ["Parch"], axis = 1)

test["FamilyNum"] = test["SibSp"] + test["Parch"]
test["hasFamily"] = test["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
test = test.drop(labels = ["SibSp"], axis = 1)
test = test.drop(labels = ["Parch"], axis = 1)

train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(train.drop("Survived", axis=1), 
                                                    train["Survived"], test_size=0.3, random_state=42)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
rsvm = SVC(kernel="rbf",probability=True, random_state=42).fit(X_train, y_train)
r0trs = round(rsvm.score(X_train,y_train), 4)
r0vals = round(rsvm.score(X_valid, y_valid), 4)
print("デフォルト")
print(f"学習データに対するスコア:{r0trs}\n検証データに対するスコア:{r0vals}")
from scipy import stats
params0 = {'C':stats.expon(scale=1000), 'gamma':stats.expon(scale=0.0001)}
random_svm = RandomizedSearchCV(SVC(kernel='rbf', random_state=42), params0, cv=10, n_jobs=-1,
                             return_train_score=False, n_iter=1000, verbose=2, scoring='accuracy',random_state=42)
random_svm = random_svm.fit(X_train, y_train)
tuned_svm = SVC(kernel='rbf', C=random_svm.best_params_['C'], gamma=random_svm.best_params_['gamma'],
                random_state=42).fit(X_train, y_train)
print("ランダムサーチにてチューニングしたベストなパラメータ")
print(random_svm.best_params_)

rtr_sc = round(tuned_svm.score(X_train, y_train), 4)
rval_sc = round(tuned_svm.score(X_valid, y_valid), 4)
print("before")
print(f"学習データに対するスコア:{r0trs}\n検証データに対するスコア:{r0vals}")
print("after")
print(f"学習データに対するスコア:{rtr_sc}\n検証データに対するスコア:{rval_sc}")

l = lambda x:10**(-x)
params2_1 = {
    'C':l(np.linspace(-4,4,9)),
    'gamma':l(np.linspace(0,5,6))
}
pprint(params2_1)
gscv2_1 = GridSearchCV(SVC(kernel='rbf',probability=True, random_state=42), params2_1, cv=5, verbose=2, n_jobs=-1, scoring='accuracy')
gscv2_1.fit(X_train, y_train)
best2_1 = gscv2_1.best_estimator_
print(f"探索結果:{best2_1}")
tr_sc2_1, val_sc2_1 = round(best2_1.score(X_train, y_train),4), round(best2_1.score(X_valid, y_valid),4)
print(f"学習データに対するスコア:{tr_sc2_1}\n検証データに対するスコア:{val_sc2_1}")
cv_result = pd.DataFrame(gscv2_1.cv_results_)
cv_result = cv_result[['param_C', 'param_gamma','mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_C', 'param_gamma')

heat_map = sns.heatmap(cv_result_pivot, cmap='Greys', annot=True)
params2_2a = {
    'C':np.linspace(500,1500,51),
    'gamma':np.linspace(0.00005,0.00015,5)
}
params2_2b = {
    'C':np.linspace(8000,12000,51),
    'gamma':np.linspace(0.00005,0.00015,5)
}
pprint(params2_2a)
print("")
pprint(params2_2b)
gscv2_2a = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), params2_2a, cv=5, verbose=2, n_jobs=-1, scoring='accuracy')
gscv2_2b = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), params2_2b, cv=5, verbose=2, n_jobs=-1, scoring='accuracy')
gscv2_2a.fit(X_train, y_train)
gscv2_2b.fit(X_train, y_train)
best2_2a = gscv2_2a.best_estimator_
tr_sc2_2a, val_sc2_2a = round(best2_2a.score(X_train, y_train),4), round(best2_2a.score(X_valid, y_valid),4)
best2_2b = gscv2_2b.best_estimator_
tr_sc2_2b, val_sc2_2b = round(best2_2b.score(X_train, y_train),4), round(best2_2b.score(X_valid, y_valid),4)
print(best2_2a)
print(f"学習データに対するスコア:{tr_sc2_2a}, 検証データに対するスコア:{val_sc2_2a}")
print(best2_2b)
print(f"学習データに対するスコア:{tr_sc2_2b}, 検証データに対するスコア:{val_sc2_2b}")
cv_result = pd.DataFrame(gscv2_2b.cv_results_)
cv_result = cv_result[['param_C', 'param_gamma','mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_C', 'param_gamma')

heat_map = sns.heatmap(cv_result_pivot, cmap='Greys', annot=False)
params2_3 = {
    'C':np.linspace(8000,14000,41),
    'gamma':np.linspace(0.000074,0.000076,11)
}
pprint(params2_3)
gscv2_3 = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), params2_3, cv=5, verbose=2, n_jobs=-1, scoring='accuracy')
gscv2_3.fit(X_train, y_train)
best2_3 = gscv2_3.best_estimator_
tr_sc2_3, val_sc2_3 = round(best2_3.score(X_train, y_train),4), round(best2_3.score(X_valid, y_valid),4)
print(best2_3)
print(f"学習データに対するスコア:{tr_sc2_3}, 検証データに対するスコア:{val_sc2_3}")
cv_result = pd.DataFrame(gscv2_3.cv_results_)
cv_result = cv_result[['param_C', 'param_gamma','mean_test_score']]
cv_result_pivot = cv_result.pivot_table('mean_test_score', 'param_C', 'param_gamma')

heat_map = sns.heatmap(cv_result_pivot, cmap='Greys', annot=False)
print("グリッドサーチによるチューニング結果")
print("チューニング前")
print("before")
print(f"学習データに対するスコア:{r0trs}\n検証データに対するスコア:{r0vals}")
print("チューニング後")
print(f"学習データに対するスコア:{tr_sc2_2b}\n検証データに対するスコア:{val_sc2_2b}")
from functools import partial

from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import space_eval

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate


def obj_func(X, y, args):
    model = SVC(kernel='rbf', random_state=42, **args)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X, y=y, cv=kf)

    return -1 * scores['test_score'].mean()

def main():
    func = partial(obj_func, X_train, y_train)
    space = {
        'C': hp.uniform('C', 0.00001, 1000),
        'gamma': hp.uniform('gamma', 0.0000001, 0.00001),
    }

    trials = Trials()
    best = fmin(fn=func, space=space, algo=tpe.suggest, max_evals=25, trials=trials, rstate=np.random.RandomState(42))
        
    print(space_eval(space, best))
    
    return best, trials

if __name__ == '__main__':
    params, trials = main()

bsvc = SVC(kernel='rbf', random_state=42, C=params['C'], gamma=params['gamma']).fit(X_train, y_train)
print("ベイズ最適化を使用した探索によるチューニング結果")
print(f"学習データに対するスコア:{round(bsvc.score(X_train, y_train),4)}")
print(f"検証データに対するスコア:{round(bsvc.score(X_valid, y_valid),4)}")
print("デフォルト")
print(f"学習データに対するスコア:{r0trs}\n検証データに対するスコア:{r0vals}\n")
print("ランダムサーチによるチューニング結果")
print(f"学習データに対するスコア:{rtr_sc}\n検証データに対するスコア:{rval_sc}\n")
print("グリッドサーチによるチューニング結果")
print(f"学習データに対するスコア:{tr_sc2_2b}\n検証データに対するスコア:{val_sc2_2b}\n")
print("ベイズ最適化を使用した探索によるチューニング結果")
print(f"学習データに対するスコア:{round(bsvc.score(X_train, y_train),4)}")
print(f"検証データに対するスコア:{round(bsvc.score(X_valid, y_valid),4)}")
plt.figure(figsize=(8,8))
for i in range(len(trials.trials)):
    his = trials.trials[i]['misc']['vals']
    plt.scatter(his['C'], his['gamma'], c='green', s=i**2, alpha=0.6)
    plt.text(his['C'][0],his['gamma'][0], i+1, size = 14, color = "black")
plt.grid()
plt.xlabel('C',size=20)
plt.ylabel('$\gamma$',size=20, rotation=1)
plt.show()
import itertools 
p_c = np.linspace(0,0.1,11)
p_g = np.linspace(0,0.1,11)
plt.figure(figsize=(8,8))
for i,j in itertools.product(p_c, p_g):
    plt.scatter(round(i,2),round(j,2), c='black')
plt.grid()
plt.xlabel('C',size=20)
plt.ylabel('$\gamma$',size=20, rotation=1)
plt.show()
bef = time.time()
lsvm = SVC(kernel="linear", random_state=42).fit(X_train, y_train)
diff = round(time.time() - bef, 4)

bef2 = time.time()
lsvm2 = LinearSVC(random_state=42).fit(X_train, y_train)
diff2 = round(time.time() - bef2, 4)
print(f"線形カーネルを使ったSVCの学習時間:{diff}秒 \nLinearSVCの学習時間:{diff2}秒")


