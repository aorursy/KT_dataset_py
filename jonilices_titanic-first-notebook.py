# Principal libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
sns.set()
# Loading the dataset

data = pd.read_csv("../input/titanic/train.csv")
data_raw = data.copy() #Just in case

data.head()
# Column names

print("The column names are:", data.columns)
# First look to the missing data

total = data.isnull().sum().sort_values(ascending = False)
porcentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, porcentage], axis = 1, keys = ["Total", "Porcentage"])
missing_data
# How many people survived plot

fig, ax = plt.subplots(1, 2, figsize = (15,5))
sns.countplot(data["Survived"], ax = ax[0])
ax[0].set_title("How many people survived?")
ax[0].set_ylabel("Count")
sns.countplot("Sex", hue = "Survived", data = data, ax = ax[1])
ax[1].set_title("Survived by Sex")
ax[1].set_ylabel("Count")

plt.show()
# Pclass analysis

fig, ax = plt.subplots(1, 2, figsize = (15,5))
sns.countplot(data["Pclass"], ax = ax[0])
ax[0].set_title("Pclass Analysis")
ax[0].set_ylabel("Count")
sns.barplot(x = "Pclass", y = "Survived", data = data, ax = ax[1])
ax[1].set_title("Survived by Pclass")
ax[1].set_ylabel("Porcentage of total")

plt.show()
# Crosstab 

pd.crosstab(data["Pclass"], data["Survived"], margins = True)
# Pivot Table

data.pivot_table("Survived", index = "Sex", columns = "Pclass")
# Survived by Sex and Pclass

fig, ax = plt.subplots(1, 2, figsize = (15,5))
sns.countplot("Pclass", hue = "Survived", data = data, ax = ax[0])
ax[0].set_title("Pclass Analysis")
ax[0].set_ylabel("Count")
sns.countplot("Sex", hue = "Pclass", data = data, ax = ax[1])
ax[1].set_title("Sex by Pclass")
ax[1].set_ylabel("Count")

plt.show()
# Crosstab 

pd.crosstab([data["Survived"], data["Sex"]], data["Pclass"], margins = True)
# Lets try to get some extra info about the age

data["Age"].describe()
# Violin and Box Plots

fig, ax = plt.subplots(1, 2, figsize = (15, 5))
sns.boxplot("Sex", "Age", hue = "Survived", data = data, ax = ax[0])
ax[0].set_title("Box Plot")
sns.violinplot("Sex", "Age", hue = "Survived", data = data, split = True, ax = ax[1])
ax[1].set_title("Violin Plot")

plt.show()
# Violin plot for Age, Pclass and Survived

fig = sns.violinplot("Pclass", "Age", hue = "Survived", split = True, data = data)
fig.set_title("Pclass and Age survirval")
plt.show()
# Extract the salutations (THANKS TO ash316)

data["Initial"] = 0
for i in data:
    data["Initial"] = data["Name"].str.extract('([A-Za-z]+)\.')
    
data.head()
# Extract all the salutations

print(data["Initial"].unique())
# Now we can replace them

data["Initial"].replace(["Mlle", "Mme", "Ms", "Dr", "Major", "Lady", "Countess",
                        "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don"], 
                        ["Miss", "Miss", "Miss", "Mr", "Mr", "Mrs", "Mrs", "Other",
                        "Other", "Other", "Mr", "Mr", "Mr"], inplace = True)

data.groupby("Initial")["Age"].mean()
# Assign the new values

data.loc[(data["Age"].isnull())&(data["Initial"]=="Mr"), "Age"] = 33
data.loc[(data["Age"].isnull())&(data["Initial"]=="Miss"), "Age"] = 22
data.loc[(data["Age"].isnull())&(data["Initial"]=="Master"), "Age"] = 5
data.loc[(data["Age"].isnull())&(data["Initial"]=="Mrs"), "Age"] = 36
data.loc[(data["Age"].isnull())&(data["Initial"]=="Other"), "Age"] = 46
# Take a look now into the missing data

total = data.isnull().sum().sort_values(ascending = False)
porcentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, porcentage], axis = 1, keys = ["Total", "Porcentage"])
missing_data
# Plot Embarked and Survival

fig, ax = plt.subplots(1, 2, figsize = (15, 5))
sns.countplot("Embarked", hue = "Survived", data = data, ax = ax[0])
ax[0].set_title("Embarked and survived")
ax[0].set_ylabel("Count")
sns.countplot("Embarked", hue = "Sex", data = data, ax = ax[1])
ax[1].set_title("Embarked by Sex")
ax[1].set_ylabel("Count")

plt.show()
# Crosstab

pd.crosstab([data["Survived"], data["Embarked"]], data["Pclass"], margins = True)
# Filling missing values

data["Embarked"].fillna("S", inplace = True)
# Take a look now into the missing data

total = data.isnull().sum().sort_values(ascending = False)
porcentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, porcentage], axis = 1, keys = ["Total", "Porcentage"])
missing_data
# SibSp plot

fig = sns.barplot("SibSp", "Survived", data = data)
fig.set_title("SibSp and Survived")

plt.show()
# SibSp plot with Pclass

fig = sns.countplot("SibSp", hue = "Pclass", data = data)
fig.set_title("Pclass with SibSp")
fig.set_ylabel("Count")

plt.show()
# A brief summary of Fare

data["Fare"].describe()
# Correlation Plot

sns.heatmap(data.corr(), annot = True, linewidths = 0.1)
plt.show()
# Removing non-relevant features

non_relevant_f = ["PassengerId", "Cabin", "Name", "Ticket", "Initial"]
data = data.drop(non_relevant_f, axis = 1)

data.head()
# Split the data

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
# Encoding categorical features

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # "Sex"
labelencoder_X_2 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6]) # "Embarked"

transformer = ColumnTransformer(
    transformers=[
        ("Titanic",
        OneHotEncoder(categories="auto"),
        [1]
        )
    ], remainder="passthrough"
)
X = transformer.fit_transform(X)
X = X[:, 1:]
# Last but not least..

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2,
                                                   random_state = 42)
# Its important to scale the data to make the model better

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
prediction_lr = model_lr.predict(X_test)

print("The accuracy of the Logistic Regression is:", metrics.accuracy_score(prediction_lr, y_test))
# Random Forests

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
prediction_rf = model_rf.predict(X_test)

print("The accuracy of the Random Forests Classifier is:", metrics.accuracy_score(prediction_rf, y_test))
# Lighgt GBM

import lightgbm as lgb
from sklearn.metrics import accuracy_score

training_data = lgb.Dataset(data = X_train, label = y_train)
params = {'num_leaves': 31, 'num_trees': 100, 'objective':'binary'}
params['metric'] = ['auc', 'binary_logloss']
classifier = lgb.train(params = params,
                      train_set = training_data,
                      num_boost_round=10)

prob_pred = classifier.predict(X_test)
y_pred=np.zeros(len(prob_pred))
for i in range(0, len(prob_pred)):
    if prob_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
accuracy = accuracy_score(y_pred, y_test) * 100
print("Accuracy: {:.0f} %".format(accuracy))
