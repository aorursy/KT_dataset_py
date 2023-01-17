%matplotlib inline

import os 

import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
train.head()
train = train.drop(["PassengerId"], axis=1)
# Before we start, let's copy the data.

train_copy = train.copy()

test_copy = test.copy()



train_copy.describe()
train_copy.info()
corr = train_copy.corr()
corr["Survived"].sort_values(ascending=False)

import seaborn as sns



colormap = sns.cubehelix_palette(light=1, as_cmap=True)

a4_dims = (10, 10)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
%matplotlib inline 

import matplotlib.pyplot as plt 

train_copy.hist(bins=50, figsize=(20,15)) 

plt.show()
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.violinplot("Survived", "Age", data=train_copy,

                   palette=["#E4421A", "#06EFA0"]);
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.violinplot("Survived", "Fare", data=train_copy,

                   palette=["#E4421A", "#06EFA0"]);
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

with sns.axes_style(style=None):

    sns.violinplot("Parch", "Age", hue="Survived", data=train_copy,

                      split=True, inner="quartile",

                      palette=["#E4421A", "#06EFA0"]);

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.violinplot("Survived", "Pclass", data=train_copy,

                   palette=["#E4421A", "#06EFA0"]);
train_copy["Cabin"].describe().top
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

with sns.axes_style(style=None):

    sns.violinplot("SibSp", "Age", hue="Survived", data=train_copy,

                      split=True, inner="quartile",

                      palette=["#E4421A", "#06EFA0"]);
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.violinplot("Sex", "Survived", data=train_copy,

                   palette=["#4F56CE", "#FF4365"]);
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

# Age by decade (for easier viewability)

train_copy['age_dec'] = train_copy.Age.map(lambda age: 10 * (age // 10))



with sns.axes_style(style=None):

    sns.violinplot("age_dec", "Survived", hue="Sex", data=train_copy,

                      split=True, inner="quartile",

                      palette=["#4F56CE", "#FF4365"]);

def return_title(x):

    title = x.split(',')[1].split('.')

    return title[0].strip()



titles_train = train_copy["Name"].transform(return_title)

titles_test = test_copy["Name"].transform(return_title)
def categorize_title(title):

    officer = ['Capt','Col','Major']

    miss = ['Miss', 'Mlle']

    mrs = ['Mrs', 'Ms', 'Mme']

    mr = ['Mr']

    other = ['Master', 'Rev', 'Dr', 'Jonkheer', 'Don', 'the Countess', 'Lady', 'Sir']

    

    if title in officer:

        return 'Officer'

    elif title in miss:

        return 'Miss'

    elif title in mrs:

        return 'Mrs'

    elif title in mr:

        return "Mr"

    elif title in other:

        return 'Other'



# We will save these for the feature engineering section 

saved_titles_train = titles_train.transform(categorize_title)

saved_titles_test = titles_test.transform(categorize_title)



saved_titles_test.unique()
# Let's take a look at what's missing here

train_copy.info()
# Let's handle the Age column first. Let's use the mean for the missing values. 

mean_age_train = np.round(train_copy["Age"].mean())

mean_age_test = np.round(test_copy["Age"].mean())



train_copy["Age"] = train_copy["Age"].fillna(mean_age_train)

test_copy["Age"] = test_copy["Age"].fillna(mean_age_test)
train_copy.info()
train_copy = train_copy.drop("Cabin",axis=1)

test_copy = test_copy.drop("Cabin",axis=1)
unknown_embarkment = train_copy[train_copy["Embarked"].isnull()]



print(unknown_embarkment["Fare"])
grid = sns.factorplot("Embarked","Fare", hue="Pclass", data=train_copy, kind="box",size=9) 

grid.axes[0][0].hlines(80,-1000,1000)

grid.set_axis_labels("Embarked","Fare");
unknown_embarkment
train_copy.set_value(61, 'Embarked', 'C')

train_copy.set_value(829,'Embarked' ,'C')
train_copy.info()
encoded_embark_train = pd.get_dummies(train_copy["Embarked"])

encoded_embark_test = pd.get_dummies(test_copy["Embarked"])



train_copy = train_copy.join(encoded_embark_train)

test_copy = test_copy.join(encoded_embark_test)



train_copy = train_copy.drop(["Embarked"], axis=1)

test_copy = test_copy.drop(["Embarked"], axis=1)
encoded_gender_train = pd.get_dummies(train_copy["Sex"])

encoded_gender_test = pd.get_dummies(test_copy["Sex"])



train_copy = train_copy.join(encoded_gender_train)

test_copy = test_copy.join(encoded_gender_test)
# Passenger + SibSP + Parch

train_copy["Family Size"] = 1 + train_copy["SibSp"] + train_copy["Parch"]

test_copy["Family Size"] = 1 + test_copy["SibSp"] + test_copy["Parch"]
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

with sns.axes_style(style=None):

    sns.violinplot("Family Size", "Age", hue="Survived", data=train_copy,

                      split=True, inner="quartile",

                      palette=["#E4421A", "#06EFA0"]);
train_copy["Title"] = saved_titles_train

test_copy["Title"] = saved_titles_test
encoded_titles_train = pd.get_dummies(train_copy["Title"])

encoded_titles_test = pd.get_dummies(test_copy["Title"])



encoded_titles_train.head()
train_copy = train_copy.join(encoded_titles_train)

test_copy = test_copy.join(encoded_titles_test)
saved_train_titles = train_copy["Title"]

saved_test_titles = test_copy["Title"]



train_copy = train_copy.drop(["Title"], axis=1)

test_copy = test_copy.drop(["Title"], axis=1)
train_copy.head()
import seaborn as sns

corr = train_copy.corr()

colormap = sns.cubehelix_palette(light=1, as_cmap=True)

a4_dims = (10, 10)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# Bring back original Title column

train_copy["Title"] = saved_titles_train

test_copy["Title"] = saved_titles_test



# Miss

train_copy[train_copy["Title"] == 'Miss']["Pclass"].value_counts()
# Mrs 

train_copy[train_copy["Title"] == 'Mrs']["Pclass"].value_counts()
train_copy[train_copy["Title"] == 'Mrs']["Family Size"].value_counts()
train_copy[train_copy["Title"] == 'Miss']["Family Size"].value_counts()
def is_child(age):

    if age < 18:

        return 1

    else:

        return 0 

    

train_copy["Child"] = train_copy["Age"].transform(is_child)

test_copy["Child"] = test_copy["Age"].transform(is_child)



train_copy["Mother"] = 0

test_copy["Mother"] = 0



train_copy.loc[(train_copy["Sex"] == 'female') & (train_copy['Age'] > 18), 'Mother'] = 1

test_copy.loc[(train_copy["Sex"] == 'female') & (train_copy['Age'] > 18), 'Mother'] = 1
duplicate_tickets_train = train_copy[train_copy.duplicated("Ticket")]

tickets_of_mothers_with_duplicate_ticket_train = list(duplicate_tickets_train [(duplicate_tickets_train["Mother"] == 1) & ( (duplicate_tickets_train["Miss"] == 1) | (duplicate_tickets_train["Mrs"] == 1) )]["Ticket"])



train_copy["Child of Mrs or Miss"] = 0 

train_copy.loc[(train_copy["Child"] == 1) & (train_copy["Ticket"].isin(tickets_of_mothers_with_duplicate_ticket_train)), "Child of Mrs or Miss"] = 1  





duplicate_tickets_test = test_copy[test_copy.duplicated("Ticket")]

tickets_of_mothers_with_duplicate_ticket_test = list(duplicate_tickets_test [(duplicate_tickets_train["Mother"] == 1) & ( (duplicate_tickets_test["Miss"] == 1) | (duplicate_tickets_test["Mrs"] == 1) )]["Ticket"])



test_copy["Child of Mrs or Miss"] = 0 

test_copy.loc[(test_copy["Child"] == 1) & (test_copy["Ticket"].isin(tickets_of_mothers_with_duplicate_ticket_test)), "Child of Mrs or Miss"] = 1    
train_copy = train_copy.drop(["Title"], axis=1)

test_copy = test_copy.drop(["Title"], axis=1)
import seaborn as sns

corr = train_copy.corr()

colormap = sns.cubehelix_palette(light=1, as_cmap=True)

a4_dims = (10, 10)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

train_copy["Single Women"] = 0 

train_copy.loc[(train_copy["Mother"] == 1) & (train_copy["SibSp"] == 0) & (train_copy["Parch"] == 0), "Single Women"] = 1



test_copy["Single Women"] = 0 

test_copy.loc[(test_copy["Mother"] == 1) & (test_copy["SibSp"] == 0) & (test_copy["Parch"] == 0), "Single Women"] = 1
import seaborn as sns

corr = train_copy.corr()

colormap = sns.cubehelix_palette(light=1, as_cmap=True)

a4_dims = (10, 10)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# Drop Uneccessary Columns 

train_copy.columns
train_copy = train_copy.drop(["Ticket", "age_dec", "Sex"], axis=1)

test_copy = test_copy.drop(["PassengerId", "Sex"], axis=1)
train_copy.columns
test_copy.columns
# Setup Training Data 

train_X = train_copy[['Pclass','male', 'female', 'Age', 'SibSp', 'Parch', 'Fare','C', 'Q', 'S', 'Family Size', 'Miss', 'Mr', 'Mrs', 'Officer', 'Other', 'Child', 'Mother', 'Child of Mrs or Miss', 'Single Women']]

test_X = test_copy[['Pclass','male', 'female', 'Age', 'SibSp', 'Parch', 'Fare','C', 'Q', 'S', 'Family Size', 'Miss', 'Mr', 'Mrs', 'Officer', 'Other', 'Child', 'Mother', 'Child of Mrs or Miss', 'Single Women']]



# Setup Target Data 

train_Y = train_copy[['Survived']]
# One 'Fare' value is NaN, let's fix that.

test_X["Fare"] = test_X["Fare"].fillna(test_X["Fare"].mean())
from sklearn.linear_model import LogisticRegression 

from sklearn import metrics

# Logistic Regression

model = LogisticRegression()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)
prediction