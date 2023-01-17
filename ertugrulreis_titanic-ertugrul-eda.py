# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



from collections import Counter

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plt.style.available
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")



test_PassengerId = test_df["PassengerId"] # PassengerId'nin ilk halini kaybetmemesini istediğimiz için
train_df.columns
train_df.head()
train_df.describe() # Numerical feature
train_df.info()
def bar_plot(variable):

    """

    input: variable ex: "Sex"

    output: bar plot & value count

    """

    # get feature 

    var = train_df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize, 

    plt.figure(figsize=(18, 5))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print(f"{variable}:\n{varValue}")
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]



for c in category1:

    bar_plot(c)
category2= ["Cabin", "Name", "Ticket"]

for c in category2:

    print(f"{train_df[c].value_counts()} \n")
def plot_hist(variable):

    

    plt.figure(figsize=(18, 6))

    

    plt.hist(train_df[variable], bins=891)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title(f"{variable} distribution with hist")

    plt.grid(True)

    

    plt.show()
category3 = ["Fare", "Age", "PassengerId"]



for c in category3:

    plot_hist(c)
# Pclass vs Survived



train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(

    by="Survived", ascending=False)
# Sex



train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(

    by="Survived", ascending=False)
#SibSp



train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(

    by="Survived", ascending=False)
# Parch



train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(

    by="Survived", ascending=False)
train_df[["Sex", "Pclass", "Survived"]].groupby(["Sex", "Pclass"], as_index=False).mean().sort_values(

    by="Survived", ascending=False)
train_df[["Sex", "Pclass", "Parch", "SibSp","Survived"]].groupby(

    ["Sex", "Pclass", "Parch", "SibSp"], as_index=False).mean().sort_values(by="Survived",

                                                                            ascending=False)[:60]


def detect_outliers(df, features):

    outlier_indices=list()

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c], 25)

        

        # 3rd quartile

        Q3 = np.percentile(df[c], 75)

        

        # IQR

        IQR = Q3 - Q1 

        

        # Outlier

        outlier_step = IQR*1.5

        

        # Detect Outlier and Indices

        outlier_detect_column = df[(df[c]<(Q1-outlier_step)) | (df[c]>(Q3+outlier_step))].index

        

        # Store Indices

        outlier_indices.extend(outlier_detect_column)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outlier = list(i for i, j in outlier_indices.items() if j>2)

    

    return multiple_outlier

train_df.loc[detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]
# Drop Outlier



train_df = train_df.drop(detect_outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis=0).reset_index(

    drop = True)
train_df_len = len(train_df)

train_df = pd.concat([train_df, test_df],axis=0).reset_index(drop=True)
train_df.info()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare", by="Embarked");
train_df["Embarked"] = train_df["Embarked"].fillna("C") 
train_df.columns[train_df.isnull().any()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))

train_df[train_df["Fare"].isnull()]
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('mode.chained_assignment', None)

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder

import pylab as pl
#Присвоим к переменным имеющиеся датасеты

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df
train_df.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True);
fig, axes = plt.subplots(ncols=2)

train_df.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')

train_df.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch');
train_df.PassengerId[train_df.Cabin.notnull()].count()
train_df.PassengerId[train_df.Age.notnull()].count()
train_df.Age = train_df.Age.median()

train_df.Age
train_df[train_df.Embarked.isnull()]
MaxPassEmbarked = train_df.groupby('Embarked').count()['PassengerId']

train_df.Embarked[train_df.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

train_df
train_df.columns[train_df.isnull().any()]
label = LabelEncoder()

dicts = {}



label.fit(train_df.Sex.drop_duplicates()) 

dicts['Sex'] = list(label.classes_)

train_df.Sex = label.transform(train_df.Sex) 



label.fit(train_df.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

train_df.Embarked = label.transform(train_df.Embarked)



train_df
test_df.Age[test_df.Age.isnull()] = test_df.Age.mean()

test_df.Fare[test_df.Fare.isnull()] = test_df.Fare.median() 

MaxPassEmbarked = test_df.groupby('Embarked').count()['PassengerId']

test_df.Embarked[test_df.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

result = pd.DataFrame(test_df.PassengerId)

test_df = test_df.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)



label.fit(dicts['Sex'])

test_df.Sex = label.transform(test_df.Sex)



label.fit(dicts['Embarked'])

test_df.Embarked = label.transform(test_df.Embarked)
test_df
target = train_df.Survived

train_df = train_df.drop(['Survived'], axis=1)

kfold = 5

itog_val = {} 
train_df
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train_df, target, test_size=0.25) 
model_rfc = RandomForestClassifier(n_estimators = 80, max_features='auto', criterion='entropy',max_depth=4) 

model_knc = KNeighborsClassifier(n_neighbors = 18) 

model_lr = LogisticRegression(penalty='l2', tol=0.01) 

model_svc = svm.SVC()
pl.clf()

plt.figure(figsize=(8,6))



#SVC

model_svc.probability = True

probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))



#RandomForestClassifier

probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))



#KNeighborsClassifier

probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))



#LogisticRegression

probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))



pl.plot([0, 1], [0, 1], 'k--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.0])

pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.legend(loc=0, fontsize='small')

pl.show()
model_rfc.fit(train_df, target)

result.insert(1,'Survived', model_rfc.predict(test_df))

result.to_csv('predictions.csv', index=False)