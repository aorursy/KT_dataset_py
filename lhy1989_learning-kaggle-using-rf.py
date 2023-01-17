# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv("../input/train.csv")

#titanic.head()

print (titanic.describe())
# 发现age有缺失

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print(titanic.describe())
# 对数值离散化

# TODO:归一化



print(titanic["Sex"].unique())



titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



titanic.head()
print(titanic["Embarked"].unique())



titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



titanic["Embarked"] = titanic["Embarked"].fillna(0)



titanic.head()
# 交叉验证，避免过拟合



from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



alg = LinearRegression()

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

#    print(train)

#    print(test)

    # 把train中predictors特征拿出来

    train_predictors = (titanic[predictors].iloc[train, :])

    # 把train y拿出来

    train_target = titanic["Survived"].iloc[train]

#    print(train_predictors.describe())

#    print(train_target.describe())

    # 查找空值，非常好用

    print(train_predictors.isnull().any())

    print(train_target.isnull().any())    

    # 计算拟合

    alg.fit(train_predictors, train_target)

    # 与test进行对比

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    # 存储输出值

    predictions.append(test_predictions)
import numpy as np



predictions = np.concatenate(predictions, axis=0)



predictions[predictions > .5] = 1

predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
# 改用logic

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression



alg = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

print(scores.mean())
# 改用随机森林

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)



kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)



print(scores.mean())
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)



kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)



print(scores.mean())
# 增加特征

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]



titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
import re



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.',name)

    

    if title_search:

        return title_search.group(1)

    return ""



titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))



title_mapping = {

    "Mr":          1,

    "Miss":        2,

    "Mrs":         3,

    "Master":      4,

    "Dr":          5,

    "Rev":         6,

    "Mlle":        7,

    "Major":       8,

    "Col":         9,

    "Mme":         10,

    "Ms":          11,

    "Capt":        12,

    "Lady":        13,

    "Jonkheer":    14,

    "Countess":    15,

    "Sir":         16,

    "Don":         17

}

for k,v in title_mapping.items():

    titles[titles == k] = v



titanic["Title"] = titles
# 检查特征的作用，给测试特征上噪音



import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]



selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



scores = -np.log10(selector.pvalues_)



plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()



predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
# 集成学习，并且去掉意义不大的特征



from sklearn.ensemble import GradientBoostingClassifier

import numpy as np



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),

    ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]],    

    [LogisticRegression(random_state=1),

    ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]]

]



predictions = []

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]

    full_test_predictions = []

    

    for alg, predictors in algorithms:

        alg.fit(titanic[predictors].iloc[train,:], train_target)

        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

        

    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2

    test_predictions[test_predictions <= .5] = 0

    test_predictions[test_predictions > .5] = 1    

    predictions.append(test_predictions)



# 改变维度    

predictions = np.concatenate(predictions, axis=0)    



accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
# 集成学习加入权重项

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),

    predictors],    

    [LogisticRegression(random_state=1),

    ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]]

]



full_test_predictions = []

for alg, predictors in algorithms:

    alg.fit(titanic[predictors].iloc[train,:], train_target)

    test_predictions = alg.predict_proba(titanic[predictors].iloc[train,:].astype(float))[:,1]

    full_test_predictions.append(test_predictions)

    

predictions = full_test_predictions[0] * .7 + full_test_predictions[1] * .3

predictions[predictions <= .5] = 0

predictions[predictions > .5] = 1 

accuracy = sum(predictions[predictions == titanic.iloc[train]["Survived"]]) / len(predictions)

print(accuracy)

print(predictions.shape)

print(titanic["Survived"].shape)