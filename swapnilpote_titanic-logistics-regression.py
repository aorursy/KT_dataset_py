import numpy as np 

import pandas as pd 



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)



import warnings

warnings.simplefilter(action='ignore')
# Read CSV train data file into DataFrame

train_df = pd.read_csv("../input/titanic/train.csv")



# Read CSV test data file into DataFrame

test_df = pd.read_csv("../input/titanic/test.csv")



# preview train data

train_df.head()
print('The number of samples into the train data is {}.'.format(train_df.shape[0]))
# preview test data

test_df.head()
print('The number of samples into the test data is {}.'.format(test_df.shape[0]))
# check missing values in train data

train_df.isnull().sum()
# percent of missing "Age" 

print('Percent of missing "Age" records is %.2f%%' % ((train_df['Age'].isnull().sum()/train_df.shape[0])*100))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df["Age"].plot(kind='density', color='teal')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
# mean age

print('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))

# median age

print('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))
# percent of missing "Cabin" 

print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))
# percent of missing "Embarked" 

print('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')

print(train_df['Embarked'].value_counts())

sns.countplot(x='Embarked', data=train_df, palette='Set2')

plt.show()
print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())
train_data = train_df.copy()

train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)
# check missing values in adjusted train data

train_data.isnull().sum()
# preview adjusted train data

train_data.head()
plt.figure(figsize=(15,8))

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df["Age"].plot(kind='density', color='teal')

ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)

train_data["Age"].plot(kind='density', color='orange')

ax.legend(['Raw Age', 'Adjusted Age'])

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
## Create categorical variable for traveling alone

train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)
train_data.head()
#create categorical variables and drop some variables

training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])

training.drop('Sex_female', axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop('Ticket', axis=1, inplace=True)



final_train = training

final_train.head()
test_df.isnull().sum()
test_data = test_df.copy()

test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)



test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)



test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)



testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])

testing.drop('Sex_female', axis=1, inplace=True)

testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)



final_test = testing

final_test.head()
plt.figure(figsize=(15,8))

ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
plt.figure(figsize=(20, 8))

avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")

plt.show()
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)



final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
plt.figure(figsize=(15,8))

ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

ax.set(xlabel='Fare')

plt.xlim(-20,200)

plt.show()
sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")

plt.show()
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")

plt.show()
sns.barplot('TravelAlone', 'Survived', data=final_train, color="mediumturquoise")

plt.show()
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

X = final_train[cols]

y = final_train['Survived']

# Build a logreg and compute the feature importances

model = LogisticRegression()

# create the RFE model and select 8 attributes

rfe = RFE(model, 8)

rfe = rfe.fit(X, y)

# summarize the selection of the attributes

print('Selected features: %s' % list(X.columns[rfe.support_]))
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)



print("Optimal number of features: %d" % rfecv.n_features_)

print('Selected features: %s' % list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 

                     'Embarked_S', 'Sex_male', 'IsMinor']

X = final_train[Selected_features]



plt.subplots(figsize=(18, 10))

sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")

plt.show()