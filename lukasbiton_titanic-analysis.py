# First, import the relevant libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# We import the data



test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
# Let's look at what the data looks like as a whole



train.head()
train.describe()
# Let's start by looking at the impact of the Pclass on survivability and how it correlates with other variables.



# Set up the matplotlib figure

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 12), sharex = False)

fig.tight_layout()



# First graph

sns.countplot(x = train.Pclass, hue = train.Survived, palette = 'pastel', ax = ax1)

ax1.set_ylabel("Count of Survivors by class")



# Second

sns.barplot(x = train.Pclass, y = train.Fare, palette = 'pastel', ax = ax2)

ax2.set_ylabel("Average Fare by class")



# Third

sns.countplot(x = train.Pclass, hue = train.Embarked, palette = 'pastel', ax = ax3)

ax3.set_ylabel("Count of Passengers by class")



# Fourth

sns.countplot(x = train.Pclass, hue = train.Sex, palette = 'pastel', ax = ax4)

ax4.set_ylabel("Count of Passengers by class")



# Fifth

sns.distplot(train[train.Pclass == 1].Age, color = 'b', ax = ax5, hist = False)

sns.distplot(train[train.Pclass == 2].Age, color = 'r', ax = ax5, hist = False)

sns.distplot(train[train.Pclass == 3].Age, color = 'g', ax = ax5, hist = False)

ax5.set_ylabel("Age of Passengers by class")



plt.show()
# Let's now look at the link between Age and survival rates.



# Set up the matplotlib figure

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20), sharex = False)

fig.tight_layout()



# First graph

sns.countplot(x = train.Sex, hue = train.Survived, palette = 'pastel', ax = ax1)

ax1.set_ylabel("Count of Survivors by Sex")



# Second

sns.barplot(x = train.Sex, y = train.Fare, palette = 'pastel', ax = ax2)

ax2.set_ylabel("Average Fare by Sex")



# Third

sns.countplot(x = train.Sex, hue = train.Embarked, palette = 'pastel', ax = ax3)

ax3.set_ylabel("Count of Passengers by Sex")



# Fourth

sns.countplot(x = train.Pclass, hue = train.Sex, palette = 'pastel', ax = ax4)

ax4.set_ylabel("Count of Passengers by Sex")



# Fifth

sns.distplot(train[train.Sex == "male"].Age, color = 'b', ax = ax5, hist = False)

sns.distplot(train[train.Sex == "female"].Age, color = 'r', ax = ax5, hist = False)

ax5.set_ylabel("Age of Passengers by Sex")



plt.show()
# Counting the number of passengers of each gender.



print("Proportion of people by gender:")

print(train.Sex.value_counts()/len(train))



print("\nSurvival rate by gender:")

print(train.groupby("Sex").mean().Survived)
# Let's now look specifically at Age and how it relates to survivability.

f, ax1 = plt.subplots(1, 1, figsize = (15, 10), sharex = False)



sns.distplot(train[train.Survived == 1].Age, ax = ax1, color = 'b', hist = False, label = "Age of Survivors")

sns.distplot(train[train.Survived == 0].Age, ax = ax1, color = 'r', hist = False, label = "Age of Non-Survivors")

plt.legend()
# We now look at the impact of having a family on surviving.



print("Number of Siblings and Spouses:")

print(train.SibSp.value_counts())

print("\nNumber of Parents and Children:")

print(train.Parch.value_counts())
# Let's visualize:



plt.figure(figsize = (5, 5))

sns.countplot(x = train.SibSp, hue = train.Survived)



# We get rid of the people with 0 SibSp to better see the other groups.

plt.figure(figsize = (10, 10))

sns.countplot(x = train[train.SibSp > 0].SibSp, hue = train.Survived)
# Let's now look at Parch



# Let's visualize:



plt.figure(figsize = (5, 5))

sns.countplot(x = train.Parch, hue = train.Survived)



# We get rid of the people with 0 SibSp to better see the other groups.

plt.figure(figsize = (10, 10))

sns.countplot(x = train[train.Parch > 0].Parch, hue = train.Survived)
# Finally we turn our attention to how Embarked impacts the survival rate



f, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10), sharex = False)



sns.barplot(x = train.Embarked, y = train.Survived, ax = ax1)

ax1.set_ylabel('Survival rate')

sns.barplot(x = train.Embarked, y = train.Fare, ax = ax2)

ax2.set_ylabel('Average Fare')
# Let's now study Fare



f, (ax1, ax2) = plt.subplots(2, 1, figsize = (15, 10), sharex = False)



sns.distplot(train[train.Survived == 1].Fare, ax = ax1, color = 'b', hist = False, label = "Fare of Survivors")

sns.distplot(train[train.Survived == 0].Fare, ax = ax1, color = 'r', hist = False, label = "Fare of Non-Survivors")

ax1.set_ylabel('Total Fare distribution')



sns.distplot(train[(train.Survived == 1) & (train.Fare < 50)].Fare, ax = ax2, color = 'b', hist = False, label = "Fare of Survivors")

sns.distplot(train[(train.Survived == 0) & (train.Fare < 50)].Fare, ax = ax2, color = 'r', hist = False, label = "Fare of Non-Survivors")

ax2.set_ylabel('Zoomed Fare Distribution')



plt.legend()
# We also check how much data is missing for both the training and testing data.



print('\nMissing data in train.\n')

print(train.isna().sum())

print('\nMissing data in test.\n')

print(test.isna().sum())
# Keep track of what gets imputed

train['Age Imputed'] = train['Age'].apply(lambda x: 1 if pd.isna(x) else 0) # I use pd.isna because it works for any object, unlike np.isnan

test['Age Imputed'] = test['Age'].apply(lambda x: 1 if pd.isna(x) else 0)



# We use class and gender because both of these seem correlated with age according

# to our graphs plotted above.

for gender in train.Sex.unique():

    for pas_class in train.Pclass.unique():

        

        # Locate by the Sex and Pclass labels the entries missing the Age, and impute using that Sex and that Pclass's data.

        

        train.loc[(train.Sex == gender) & (train.Pclass == pas_class) & (train.Age.isnull()), 'Age'] = train.groupby(by = ['Sex', 'Pclass']).mean().loc[gender, pas_class].Age

            

        test.loc[(test.Sex == gender) & (test.Pclass == pas_class) & (test.Age.isnull()), 'Age'] = train.groupby(by = ['Sex', 'Pclass']).mean().loc[gender, pas_class].Age # I use the train data to avoid data leakage
# We now impute Embarked. Looking at our earlier graph,

# We can use the fare to guess where these passengers embarked.



# Keep track of imputations

train['Embarked Imputed'] = train['Embarked'].apply(lambda x: 1 if pd.isna(x) else 0)

test['Embarked Imputed'] = test['Embarked'].apply(lambda x: 1 if pd.isna(x) else 0)



# We use the average fare for each port.

# We make a small function for readability:



def impute_embarked(fare):

    if fare > train.groupby(by = 'Embarked').mean().Fare[0]:

        return "C"

    elif fare > train.groupby(by = 'Embarked').mean().Fare[2]:

        return "Q"

    else:

        return "S"



train.loc[train.Embarked.isnull(), 'Embarked'] = train.loc[train.Embarked.isnull(), 'Fare'].apply(impute_embarked)

test.loc[test.Embarked.isnull(), 'Embarked'] = test.loc[test.Embarked.isnull(), 'Fare'].apply(impute_embarked) # Again we impute using only training data.
# Finally we impute the missing Fare values.

train['Fare Imputed'] = train['Fare'].apply(lambda x: 1 if pd.isna(x) else 0)

test['Fare Imputed'] = test['Fare'].apply(lambda x: 1 if pd.isna(x) else 0)



# We use the average fare by Pclass and Sex

for gender in train.Sex.unique():

    for pas_class in train.Pclass.unique():

        

        # Locate by the Sex and Pclass labels the entries missing the Fare, and impute using that Sex and that Pclass's data.

        

        train.loc[(train.Sex == gender) & (train.Pclass == pas_class) & (train.Fare.isnull()), 'Fare'] = train.groupby(by = ['Sex', 'Pclass']).mean().loc[gender, pas_class].Fare

            

        test.loc[(test.Sex == gender) & (test.Pclass == pas_class) & (test.Fare.isnull()), 'Fare'] = train.groupby(by = ['Sex', 'Pclass']).mean().loc[gender, pas_class].Fare # I use the train data to avoid data leakage
# Besides cabins, there is no more missing data!



print('\nMissing data in train.\n')

print(train.isna().sum())

print('\nMissing data in test.\n')

print(test.isna().sum())
# I now combine the three columns tracking imputed data to make a single column.



train['Imputed'] = train['Age Imputed'] + train['Embarked Imputed'] + train['Fare Imputed']

test['Imputed'] = test['Age Imputed'] + test['Embarked Imputed'] + test['Fare Imputed']



train.Imputed = train.Imputed.apply(lambda x: 1 if x > 0 else 0)

test.Imputed = test.Imputed.apply(lambda x: 1 if x > 0 else 0)



train.drop(['Age Imputed', 'Embarked Imputed', 'Fare Imputed'], axis = 1, inplace = True)

test.drop(['Age Imputed', 'Embarked Imputed', 'Fare Imputed'], axis = 1, inplace = True)
# We start by extracting the title from the name.

# Let's see what names look like.



train.Name.head()
# We can use the default space as a separator to only grab the title.

train['Title'] = train.Name.apply(lambda x: x.split()[1])

test['Title'] = test.Name.apply(lambda x: x.split()[1])



# Make a dictionary to replace the title values

titles = dict([('Mr.', 1), ('Mrs.', 2), ('Miss.', 3),

               ('Master.', 4), ('Planke,', 4), ('Don.', 4),

               ('Rev.', 4), ('Billiard,', 4), ('der', 4),

               ('Walle,', 4), ('Dr.', 4), ('Pelsmaeker,', 4),

               ('Mulder,', 4), ('y', 4), ('Steen,', 4),

               ('Carlo,', 4), ('Mme.', 4), ('Impe,', 4),

               ('Ms.', 4), ('Major.', 4), ('Gordon,', 4),

               ('Messemaeker,', 4), ('Mlle.', 4), ('Col.', 4),

               ('Capt.', 4), ('Velde,', 4), ('the', 4),

               ('Shawah,', 4), ('Jonkheer.', 4), ('Melkebeke,', 4),

               ('Cruyssen,', 4), ('Carlo', 4), ('Khalil', 4),

               ('y', 4), ('Palmquist,', 4), ('Brito,', 4),

               ('Khalil,', 4)])



# Replace the title data

train.Title = train.Title.replace(to_replace = titles)

test.Title = test.Title.replace(to_replace = titles)
# We now create bins for the age categories using our earlier data exploration.

# Recall that 0 to 15 seemed more likely to survive, 15 to 30 more likely not

# to survive, 30 to 57 was equally likely to survive, and above 57 again less likely to survive.

train['AgeGroup'] = train.Age.apply(lambda x: 0 if x < 15 else 3 if x > 57 else 1 if x < 30 else 2)

test['AgeGroup'] = test.Age.apply(lambda x: 0 if x < 15 else 3 if x > 57 else 1 if x < 30 else 2)
train.head()
# We now turn our attention to SibSp.

# Using our graphical analysis:



train['FamilySib'] = train.SibSp.apply(lambda x: 0 if x == 0 else 2 if x > 2 else 1)

test['FamilySib'] = test.SibSp.apply(lambda x: 0 if x == 0 else 2 if x > 2 else 1)
# And similarly for Parch:



train['FamilyPar'] = train.Parch.apply(lambda x: 0 if x == 0 else 2 if x > 3 else 1)

test['FamilyPar'] = test.Parch.apply(lambda x: 0 if x == 0 else 2 if x > 3 else 1)
# Let's now bin the fares into two categories: below 15 and above 15:



train['ExpensiveFare'] = train.Fare.apply(lambda x: 1 if x > 15 else 0)

test['ExpensiveFare'] = test.Fare.apply(lambda x: 1 if x > 15 else 0)
# Finally, we look at the cabins.



# This is to create bins for each cabin letter, which allows for one final imputation later on.

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}



def get_cabin_letter(cabin):

    if not(pd.isnull(cabin)): # This check did not work in a lambda function. It makes sure I don't try to slice a nan value.

        return cabin_mapping[cabin[0]]



train['CabinBin'] = train.Cabin.apply(get_cabin_letter)

test['CabinBin'] = test.Cabin.apply(get_cabin_letter)



# And impute the missing cabin data

train["CabinBin"].fillna(train.groupby("Pclass")["CabinBin"].transform("median"), inplace=True)

test["CabinBin"].fillna(test.groupby("Pclass")["CabinBin"].transform("median"), inplace=True)
# The motivation for using the cabin comes from the following infographics,

# which shows that some third classes and some first classes were more at

# risk than others because they sank earlier.



from IPython.display import Image

from IPython.core.display import HTML

Image(url= "https://images.squarespace-cdn.com/content/v1/5006453fe4b09ef2252ba068/1352009121819-Y8KWSMENX8KXXKKY32M9/ke17ZwdGBToddI8pDm48kDJ9qXzk7iAyoryvGQVNA_F7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UcJGkCP2eEqMWqTLP4RLEZaxwnphfgit-_H6Nuosp50rm0sD-Bab7E9MY8W31A7zMQ/TItanic-Survival-Infographic.jpg?format=2500w")
# We transform Sex and Embarked into dummy variables

train['Sex'] = train.Sex.apply(lambda x: 0 if x == 'male' else 1)

test['Sex'] = test.Sex.apply(lambda x: 0 if x == 'male' else 1)



train = pd.get_dummies(data = train, columns = ['Embarked'], drop_first = True)

test = pd.get_dummies(data = test, columns = ['Embarked'], drop_first = True)
# Finally, we can drop: name, Age, SibSp, Parch, Ticket, Fare, Cabin.

# We make dummies out of the remaining columns.



X = train.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Age', 'SibSp',

                             'Parch', 'Ticket', 'Fare', 'Cabin'], axis = 1)

y = train.Survived



test_features = test.drop(['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp',

                             'Parch', 'Ticket', 'Fare', 'Cabin'], axis = 1)
X.head()
# First we split our training data into a training set and testing set. We will then

# try out different models and use the one with the best results.

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Let's start with a logistic regression

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
log_predictions = logmodel.predict(X_test)
print(classification_report(y_test,log_predictions))

print('\n')

print(confusion_matrix(y_test,log_predictions))
# Let's now try KNN

from sklearn.neighbors import KNeighborsClassifier
# We look for the best neighbor right away.

error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn_final = KNeighborsClassifier(n_neighbors = 25)
knn_final.fit(X_train, y_train)
knn_predictions = knn_final.predict(X_test)
print(classification_report(y_test,knn_predictions))

print('\n')

print(confusion_matrix(y_test,knn_predictions))
# We now try using a random forest classifier.



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test,rfc_predictions))

print('\n')

print(confusion_matrix(y_test,rfc_predictions))
final_model = KNeighborsClassifier(n_neighbors = 25)
# Fit the model

final_model.fit(X, y)
# Make predictions

final_predictions = final_model.predict(test_features)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_predictions})
submission.to_csv('SubmissionFinal10032020.csv', index = False)