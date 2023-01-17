import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression



#get titanic and test csv files and insert into Dataframes

titanic_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

sample_submission_df = pd.read_csv("../input/gender_submission.csv")



#titanic_df.head(50)



first_class_titanic_df = titanic_df[titanic_df['Pclass'] == 3]

first_class_titanic_df.head(25)
test_df.head(5)
titanic_df['Pclass'].value_counts()

#print(count_nan_cabin_titanic)

#Drop PassengerId, Name and Ticket as not useful

titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1)

test_df = test_df.drop(['Name', 'Ticket', 'Embarked'], axis=1)
#Create plots to investigate age distribution on Titanic

fig, (axis1, axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titianic')



# Plot of ages in original form - but titanic_df has 177 NA's (out of 714 records in total).  

# Test_df has 86 NA's of 332.

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



#There are a lot of NA's.  Replace these with random values between (mean - std) and (mean + std)



#get average, std for titanic_df and test_df

average_age_titanic = titanic_df["Age"].mean()

std_age_titanic = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



average_age_test = test_df["Age"].mean()

std_age_test = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



#generate random numbers to replace NaN's with

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



#fill NaN values in Age column with random values generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2



#convert from float to int

titanic_df["Age"] = titanic_df["Age"].astype(int)

test_df["Age"] = test_df["Age"].astype(int)



#Plot amended age values 

titanic_df['Age'].hist(bins=70, ax=axis2)
#Plot of age vs survival shows that children <15 yo have higher rates of survival 



facet = sns.FacetGrid(titanic_df, hue = "Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, titanic_df['Age'].max()))

facet.add_legend()
# Children have high rates of survival.  Create new column 'Person' which classifies

# passengers as male, female or child.

def get_person(passenger):

    age,sex,pclass = passenger

    if (pclass == 1 or pclass == 2) and age < 16:

        return 'rich_child'

    elif (pclass == 1 or pclass == 2) and sex == 'female':

        return 'rich_female'    

    elif age < 16:

        return 'child'

    elif sex == 'female':

        return 'female'

    else:

        return 'male'

        

titanic_df['Person'] = titanic_df[['Age', 'Sex', 'Pclass']].apply(get_person,axis=1)

test_df['Person'] = test_df[['Age', 'Sex', 'Pclass']].apply(get_person,axis=1)



# Drop Sex column - no longer needed

titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)
titanic_df.head(40)
# Look at survival rates for women and children vs men by class.



fig, axes = plt.subplots(2,2,figsize=(15,10))



sns.countplot(x='Person', data=titanic_df, ax=axes[0,0])



person_perc = titanic_df[["Person", "Survived"]].groupby(["Person"], as_index=False).mean()

sns.barplot(x="Person", y="Survived", data=person_perc, ax=axes[0,1], order=['male', 'female', 'child'])

sns.countplot(x='Pclass', hue='Person', data=titanic_df, ax=axes[1,0])

sns.barplot(x='Pclass', y="Survived", hue='Person', data=titanic_df, ax=axes[1,1])



#Create dummy variable for Person column & drop male (as child & female columns imply whether male)



person_dummies_titanic = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Rich_Child', 'Rich_Female','Child', 'Female', 'Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Rich_Child', 'Rich_Female','Child', 'Female', 'Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df = test_df.join(person_dummies_test)



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)



titanic_df.head(5)
# PASSENGER CLASS (Pclass)



#Plot Pclass vs chance of survival

sns.factorplot('Pclass', 'Survived', order=[1,2,3], data=titanic_df, size=5)



#Create dummies for Pclass and then drop third class

pclass_dummies_titanic = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



titanic_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df = test_df.join(pclass_dummies_titanic)

titanic_df.head(5)
#CABIN



# Replace NaN with 0.

# count_nan_cabin_titanic = titanic_df['Cabin'].isnull().sum()

# print(count_nan_cabin_titanic)

titanic_df['Cabin'].fillna(value=0, inplace=True)

test_df['Cabin'].fillna(value=0, inplace=True)

titanic_df['top_cabin'] = titanic_df['Cabin'].astype(str).str[0]

sns.factorplot('top_cabin', 'Survived', data=titanic_df, size=5)

#sns.countplot(x='top_cabin', data=titanic_df)



a_deck_df = titanic_df[titanic_df['top_cabin'] == 'A']

a_deck_df.head(40)


# Passengers who had cabins at the top of the ship may have been more likely to survive than

# those who were further below.  Create column for top_cabin Location.  If passenger had cabin 

# at top of ship (ie decks A, B or C) give value 1, else 0. 



titanic_df['top_cabin'] = titanic_df['Cabin'].astype(str).str[0]

titanic_df['top_cabin'] = titanic_df['top_cabin'].map({'A': 1, 'B': 1, 'C': 1})

titanic_df['top_cabin'][titanic_df['top_cabin'] != 1] = 0

titanic_df['top_cabin'] = titanic_df['top_cabin'].astype(np.int64)



test_df['top_cabin'] = test_df['Cabin'].astype(str).str[0]

test_df['top_cabin'] = test_df['top_cabin'].map({'A': 1, 'B': 1, 'C': 1})

test_df['top_cabin'][test_df['top_cabin'] != 1] = 0

test_df['top_cabin'] = test_df['top_cabin'].astype(np.int64)



titanic_df.drop(['Cabin'],axis=1,inplace=True)

test_df.drop(['Cabin'],axis=1,inplace=True)



titanic_df.drop(['top_cabin'],axis=1,inplace=True)

test_df.drop(['top_cabin'],axis=1,inplace=True)
titanic_df.head(5)
#FAMILY



# We have seen there was a v large number of men in 3rd class.  

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.pointplot('Parch', 'Survived', data=titanic_df, ax=axis1)

sns.pointplot('SibSp', 'Survived', data=titanic_df, ax=axis2)



#sns.factorplot('Parch', data=titanic_df, ax=axis1)

# sns.factorplot('SibSp', 'Survived', data=titanic_df, ax=axis2)
#Husbands in first or second class with wife on board



# Expect that husbands may have been allowed on board to accompany their wives.



def is_husband(passenger):

    child, female, class_1, class_2, sibsp = passenger

    if (child == 0 and female == 0) and (class_1 == 1) and sibsp >= 1:

        return 1

    else: 

        return 0



titanic_df['Rich_man_travelling_w_wife'] = titanic_df[['Child', 'Female','Class_1', 'Class_2', 'SibSp']].apply(is_husband,axis=1)

test_df['Rich_man_travelling_w_wife'] = test_df[['Child', 'Female','Class_1', 'Class_2','SibSp']].apply(is_husband,axis=1)
sns.pointplot('Rich_man_travelling_w_wife', 'Survived', data=titanic_df)
titanic_df.head(5)
#Find parents with children onboard



# Expect that children were favoured and parents of those children may have been more likely 

# to survive.  Create column child_or_accomp which is 1 if parch >= 1 and 0 otherwise.



def get_parent_of_child(passenger):

    parch,age = passenger

    return 1 if age > 18 and parch >= 1 else 0



titanic_df['parent_of_child'] = titanic_df[['Parch', 'Age']].apply(get_parent_of_child,axis=1)

test_df['parent_of_child'] = test_df[['Parch', 'Age']].apply(get_parent_of_child,axis=1)



#Drop 

titanic_df.drop(['Parch'],axis=1,inplace=True)

titanic_df.drop(['SibSp'],axis=1,inplace=True)

test_df.drop(['Parch'],axis=1,inplace=True)

test_df.drop(['SibSp'],axis=1,inplace=True)



titanic_df.drop(['parent_of_child'],axis=1,inplace=True)

test_df.drop(['parent_of_child'],axis=1,inplace=True)
#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



#sns.pointplot('parent_of_child', 'Survived', data=titanic_df, ax=axis1)
#FARE



#One missing value in test_df - fill with median value

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



titanic_df["Fare"] = titanic_df['Fare'].astype(int)

test_df["Fare"] = test_df['Fare'].astype(int)
titanic_df.head(5)
X_train = titanic_df.drop("Survived", axis=1)

Y_train = titanic_df["Survived"]

X_test = test_df.drop("PassengerId", axis=1).copy()
X_test.head(5)
#Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
#Get Correlation Coefficient for each feature using logistic regression



coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df['Coefficient Estimate'] = pd.Series(logreg.coef_[0])



coeff_df



Submission = pd.DataFrame({"PassengerId":test_df["PassengerId"], "Survived": Y_pred })

Submission.to_csv('titanic.csv', index=False)