# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# lesson

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter # Next libriary

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
a = [1, 2, 3, 4, 5]

plt.plot(a)

plt.show()
plt.style.available
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()

# Lesson

test_PassengerId = test_data["PassengerId"]
print('Shape of train csv: \t', train_data.shape)

print('Shape of test csv: \t', test_data.shape)

# Lesson

train_data.columns
test_data.head()
train_data.info()
def bar_plot(variable):

    """

    input: variable ex: "age"

    output: bar plot & value count

    """

    var = train_data[variable]

    

    varValue = var.value_counts()

    

    plt.figure(figsize = (9, 3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable, varValue))
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]

for c in category1:

    bar_plot(c)
category2 = ["Cabin", "Name", "Ticket"]

for n in category2:

    print("{} \n".format(train_data[n].value_counts()))
def plot_hist(variable):

    """

    input: variable ex: "Fare"

    output: histogram plot & value counts

    """

    plt.figure(figsize = (9, 3))

    plt.hist(train_data[variable], bins = 20)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} dist with histogram".format(variable))

    plt.show()

    print("{}: \n {}".format(variable, train_data[variable].value_counts()))
numericValue = ["Fare", "Age", "PassengerId"]

for n in numericValue:

    plot_hist(n)
train_data.columns
train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean()
# Pcall for survived

train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# Sex for survived

train_data[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# Sibsp for survived

train_data[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending = False)
# Parch for survived

train_data[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending = False)
def det_outliers(data, features):

    outliers_in = []

    

    for c in features:

        # Firs Q

        Q1 = np.percentile(data[c], 25)

        # Third Q

        Q3 = np.percentile(data[c], 75)

        # IQR

        IQR = Q3 - Q1

        # outlier_step

        outlier_step = IQR * 1.5

        # detect outliers and indices

        outlier_list_col = data[(data[c] < Q1 - outlier_step) | (data[c] > Q3 + outlier_step)].index

        outliers_in.extend(outlier_list_col)

        

    outliers_in = Counter(outliers_in)

    multiple_outliers = list(i for i, v in outliers_in.items() if v > 2)

    

    return multiple_outliers
# Outlier datalarni chiqarish

train_data.loc[det_outliers(train_data, ["Age", "SibSp", "Fare", "Parch"])]
train_length = len(train_data)

all_data = pd.concat([train_data, test_data], axis=0).reset_index(drop = True)
all_data.head()
all_data.columns[all_data.isnull().any()]
all_data.isnull().sum()
all_data[all_data["Embarked"].isnull()]
all_data.boxplot(column="Fare", by="Embarked")

plt.show()
all_data["Embarked"] = all_data["Embarked"].fillna("C")
print(all_data[all_data["Embarked"].isnull()])

print(all_data.isnull().sum())
all_data[all_data["Fare"].isnull()]
all_data["Fare"] = all_data["Fare"].fillna(np.mean(all_data[all_data["Pclass"] == 3]["Fare"]))
print(all_data[all_data["Fare"].isnull()], "\n")

print(all_data.isnull().sum())
np.mean(all_data[all_data["Pclass"] == 3]["Fare"])
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# survived female %

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)

# survived male %

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)