import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import pandas_profiling as pdp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# train_df.info()
# f, ax = plt.subplots(figsize = (12, 9))
# sns.heatmap(train_df.corr(), annot = True)
# 1.Raw data
train_df.head(10)

# 2.Preprocessing
print(train_df.isnull().sum())
# 2.Preprocessing
train_df.Age = train_df.Age.fillna(train_df.Age.median)
# 2.Preprocessing
train_df.Embarked = train_df.Embarked.fillna("S")
# 2.Preprocessing
print(train_df.isnull().sum())

# temp = train_df.sort_values('Survived')

temp2 = []
temp2 = temp[temp.Survived == 0].Pclass.dropna(), temp[temp.Survived == 1].Pclass.dropna()
plt.hist(temp2, histtype="barstacked", bins=10)

temp3 = []
temp3 = temp[temp.Survived == 0].Age.dropna(), temp[temp.Survived == 1].Age.dropna()
plt.hist(temp3, histtype="barstacked", bins=10)

train_df.Sex[train_df.Sex == 'male'] = 0
train_df["Sex"][train_df["Sex"] == "female"] = 1
train_df["Embarked"][train_df["Embarked"] == "S" ] = 0
train_df["Embarked"][train_df["Embarked"] == "C" ] = 1
train_df["Embarked"][train_df["Embarked"] == "Q"] = 2


train_df.head()

f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(train_df.corr(), annot = True)
train_df.head()
from sklearn import tree
# Cooking here ------------------------------------
# Cooking below ------------------------------------
# 3.Choose variables
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(train_df.corr(), annot = True)
# 3.Choose variables
df = train_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Embarked', 'Sex', 'Age', 'Fare'], axis = 1)
dfTest = test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Embarked', 'Sex', 'Age', 'Fare'], axis = 1)
# 3.Choose variables
dfTest.head()
# 3.Choose variables
dfX = df.drop('Survived', axis=1)
# 3.Choose variables
dfY = df.Survived
# 4.Machine learning model
clf = DecisionTreeClassifier(random_state=0)

# machine learning model eats a pot!
clf = clf.fit(dfX, dfY)
pred = clf.predict(dfTest)

# 4.Machine learning model
PassengerId = np.array(test_df["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])




# Extra ------------------------------

split_data = []
for survived in [0,1]:
    split_data.append(train_df[train_df.Survived==survived])

temp2 = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp2, histtype="barstacked", bins=3)

temp2



