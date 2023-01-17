#call libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import pylab

import math

import seaborn as sns



# Set default matplot figure size

pylab.rcParams['figure.figsize'] = (6.5, 5.0)



#Turn off pandas warning for changing variables & future warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None
#set random seed

np.random.seed(123)
#Import dataset and look at what information we are dealing with

titanic = pd.read_csv("../input/train.csv", header = 0)

titanic.head(20)

#column with names all passangers

names = titanic["Name"]



#Check whether there are duplicates in the name list

duplicates = names[names.duplicated()]

print(duplicates)
titanic = titanic.drop(columns = "PassengerId")

titanic = titanic.drop(columns = "Ticket")

titanic = titanic.drop(columns = "Name")

titanic.head()
titanic.columns = (['Survived', 'Class', 'Sex', 'Age', 

                    'n_Siblings_Spouse', 'n_Parents_Chidren', 

                    'Fare_Price', 'Cabin_ID', 'Embarked'])
print("Summary statistics of the numerical columns:")

print()

print(titanic.describe())

print()

print("Missing values per column:")

print()

print(titanic.isna().sum())
# Turn sex, class, survived and embarked in categorical variable

titanic["Sex"] = titanic.Sex.astype('category')

titanic["Class"] = titanic.Class.astype('category')

titanic["Survived"] = titanic.Survived.astype('category')

titanic["Embarked"] = titanic.Embarked.astype('category')



#Rename Embarked cities

titanic["Embarked"] = titanic.Embarked.cat.rename_categories({"S" :"Southampton",

                                                              "C" : "Cherbourg",

                                                              "Q" : "Queenstown"})

#Plot barplots for the independent categorical variables 

sns.countplot(titanic.Sex).set_title('Distribution of sexes')

plt.show()

sns.countplot(titanic.Class).set_title('Distribution of classes')

plt.show()

sns.countplot(titanic.Embarked).set_title('Distribution of where people embarked')

plt.show()
#Plot the distribution of the ticket price

sns.distplot(titanic.Fare_Price, bins  = 50).set_title('Distribution of ticket prices')

plt.show()



#Plot the distribution of ages

#Since there are Na's in the age distribution, we specify we only want to see the distribution of the available age data

sns.distplot(titanic.Age[-titanic.Age.isnull()], bins = 40).set_title('Distribution of ages')

plt.show()
babies = titanic[titanic.Age < 2]

babies
#specify the ages that should be turned into 0 and change types to integer

titanic.Age[titanic.Age < 1] = 0

titanic.Age[-titanic.Age.isnull()] = titanic.Age[-titanic.Age.isnull()].astype('int')
#show observations where fare price > 100,-

titanic[titanic.Fare_Price > 100]
#Create the dataframe for ticket prices that were above 100,-

the_wealthy = titanic[titanic.Fare_Price > 100]



#calculate the percentage that was either female and the percentage that embarked in Cherbourg within this dataframe. 

print("Of the people that had a ticket price of more than 100,",

      (len(the_wealthy[the_wealthy.Sex == 'female'])) / (len(the_wealthy))*100, 

      "% was female.")

print()

print("Of the people that had a ticket price of more than 100,",

      (len(the_wealthy[the_wealthy.Embarked == 'Cherbourg'])) / (len(the_wealthy))*100,

     "% embarked in Cherbourg.")
NaN_ages = titanic[-(titanic.Age > -2)]



#Plot barplots for the independent categorical variables 

sns.countplot(NaN_ages.Sex).set_title('Distribution of sexes ageless')

plt.show()

sns.countplot(NaN_ages.Class).set_title('Distribution of classes ageless')

plt.show()

sns.countplot(NaN_ages.Embarked).set_title('Distribution of where people embarked ageless')

plt.show()

sns.countplot(NaN_ages.Survived).set_title('Distribution of whether people survived ageless')

plt.show()
#full dataset children/parent distribution

sns.countplot(titanic.n_Parents_Chidren).set_title('Children/parents amongst titanic passengers')

plt.show()

#ageless children/parent distribution

sns.countplot(NaN_ages.n_Parents_Chidren).set_title('Children/parents amongst ageless titanic passengers ')

plt.show()



#full dataset sibling/spouse distribution

sns.countplot(titanic.n_Siblings_Spouse).set_title('Siblings/spouse amongst titanic passengers')

plt.show()

#ageless sibling/spouse distribution

sns.countplot(NaN_ages.n_Siblings_Spouse).set_title('Siblings/spouse amongst ageless titanic passengers')

plt.show()
#distinguish survival by sex

sns.catplot('Sex', data = titanic, hue = 'Survived', kind='count', aspect=1.5)

plt.show()
#distinguish survival by class

sns.catplot('Class', data = titanic, hue = 'Survived', kind='count', aspect=1.5)

plt.show()
#distinguish survival by where people were embarked

sns.catplot('Embarked', data = titanic, hue = 'Survived', kind='count', aspect=1.5)

plt.show()
#make a categorical variable for the ages

titanic.loc[(titanic.Age < 15), "AgeCat"] = "Kids"

titanic.loc[(titanic.Age >= 15) & (titanic.Age <= 30), "AgeCat"] = "Adolescents"

titanic.loc[(titanic.Age >= 31) & (titanic.Age <= 60), "AgeCat"] = "Adults"

titanic.loc[(titanic.Age >= 61), "AgeCat"] = "Elderly"
#distinguish survival by agecategories



sns.catplot('AgeCat', data = titanic, hue = 'Survived', kind='count', aspect=1.5)

plt.show()
#create dataframe for only the kid category

save_the_kids = titanic[titanic.AgeCat == "Kids"]

#plot the kids' sex and survival

sns.catplot('Sex', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)

plt.show()

#plot the kids' ticket class and survival

sns.catplot('Class', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)

plt.show()

#plot the how many parents were with the children and their survival

sns.catplot('n_Parents_Chidren', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)

plt.show()

#convert all categories into numerical variables so they can be proberly used to model with

titanic.Sex = pd.CategoricalIndex(titanic.Sex)

titanic.Class = pd.CategoricalIndex(titanic.Class)

titanic.Embarked = pd.CategoricalIndex(titanic.Embarked)





titanic['Sex'] = titanic.Sex.cat.codes

titanic['Class'] = titanic.Class.cat.codes

titanic['Embarked'] = titanic.Embarked.cat.codes



titanic = titanic.drop(["Cabin_ID", "AgeCat"], axis = 1)
titanic2 = titanic.dropna()
#Split the data in a train and testset

titanic_dep = titanic2.Survived

titanic_indep = titanic2.drop(['Survived'], axis=1)



from sklearn import preprocessing

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(titanic_indep, titanic_dep, test_size=0.3)



#Tree packages for checking the feature importance

from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier() 



# Build a forest and compute the feature importances

## Fit the model on your training data.

rf.fit(X_train, y_train) 

## And score it on your testing data.

rf.score(X_test, y_test)

feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',

                                                                        ascending=False)

print(feature_importances)
#oerform logistic regression and KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



# Do a Kfold cross validation on the training data for k = 3



knn = KNeighborsClassifier(n_neighbors = 3)

CVscores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = "accuracy")

print("knn score:", CVscores.mean())



# Do a Kfold cross validation on the training data for a logistic regression



logregression = LogisticRegression(solver='liblinear')

CVscores = cross_val_score(logregression, X_train, y_train, cv = 10, scoring = "accuracy")

print("logistic regression score:", CVscores.mean())
#Import dataset and look at what information we are dealing with

titanic = pd.read_csv("../input/train.csv", header = 0)
#Create dummy variable for married by looping over whether passangers names' contain Mr. or Mrs.

titanic["Mr."] = 0

for i in range(0,len(titanic["Name"])):

    if "Mr." in titanic.loc[i]["Name"]:

        titanic.at[i, "Mr."] = 1



titanic["Mrs."] = 0

for i in range(0,len(titanic["Name"])):

    if "Mrs." in titanic.loc[i]["Name"]:

        titanic.at[i, "Mrs."] = 1

        

titanic["Miss."] = 0

for i in range(0,len(titanic["Name"])):

    if "Miss." in titanic.loc[i]["Name"]:

        titanic.at[i, "Miss."] = 1        



                

titanic["Master."] = 0

for i in range(0,len(titanic["Name"])):

    if "Master." in titanic.loc[i]["Name"]:

        titanic.at[i, "Master."] = 1

        

titanic["Other_Title"] = 1 - (titanic["Master."] + titanic["Miss."] + titanic["Mrs."] + titanic["Mr."])

#Sort by ticket number and see if we can find any interesting patterns

titanic.sort_values(by = "Ticket")
#Display the passangers that didn't pay for their tickets

titanic.loc[(titanic.Fare == 0)]
#generate a zero ticket fare variable

titanic["Zero_ticket_fare"] = 0

for i in range(0,len(titanic["Fare"])):

    if titanic.loc[i]["Fare"] == 0:

        titanic.at[i, "Zero_ticket_fare"] = 1





#locate and change ticket nr.

titanic.loc[(titanic.Ticket == "LINE"), "Ticket"] = str(370160)
#Create a seperate dataframe with count data for how often each ticket occurs

ticket_counts = titanic['Ticket'].value_counts()

ticket_counts = pd.Series.to_frame(ticket_counts)

ticket_counts["Ticket_nr"] = ticket_counts.index

ticket_counts.index = range(0,len(ticket_counts))

ticket_counts.columns = ["Ticket_group_size", "Ticket"]



#Add this column to the full dataframe 

titanic = pd.merge(titanic, ticket_counts, how='outer', on='Ticket')
#Now calculate the actual ticket value

titanic["Price_per_person"] = (titanic["Fare"] / titanic["Ticket_group_size"])
#generate dummies for the class variable

class_dummies = pd.get_dummies(titanic.Pclass)

class_dummies.columns = ["First_class", "Second_class", "Third_class"]

titanic = pd.concat([titanic, class_dummies], axis=1, sort=False)
#generate dummies for where the ship embarked

embarked_dummies = pd.get_dummies(titanic.Embarked)

embarked_dummies.columns = ["Southampton", "Cherbourg", "Queenstown"]

titanic = pd.concat([titanic, embarked_dummies], axis=1, sort=False)
#Drop columns that wont be used in the analysis

titanic = titanic.drop(columns = "PassengerId")

titanic = titanic.drop(columns = "Ticket")

titanic = titanic.drop(columns = "Name")

titanic = titanic.drop(columns = "Cabin")

titanic = titanic.drop(columns = "Fare")

titanic = titanic.drop(columns = "Pclass")

titanic = titanic.drop(columns = "Embarked")
#Seperate columns

titanic_indep = titanic.drop(columns = "Survived")

titanic_dep = titanic["Survived"]
#Change the type of all dummies to categories 

titanic_indep["Sex"] = titanic_indep.Sex.astype('category')

titanic_indep["First_class"] = titanic_indep.First_class.astype('category')

titanic_indep["Second_class"] = titanic_indep.Second_class.astype('category')

titanic_indep["Third_class"] = titanic_indep.Third_class.astype('category')

titanic_indep["Southampton"] = titanic_indep.Southampton.astype('category')

titanic_indep["Cherbourg"] = titanic_indep.Cherbourg.astype('category')

titanic_indep["Queenstown"] = titanic_indep.Queenstown.astype('category')



#convert all categories into numerical variables so they can be properly used to model with

titanic_indep.Sex = pd.CategoricalIndex(titanic_indep.Sex)

titanic_indep.First_class = pd.CategoricalIndex(titanic_indep.First_class)

titanic_indep.Second_class = pd.CategoricalIndex(titanic_indep.Second_class)

titanic_indep.Third_class = pd.CategoricalIndex(titanic_indep.Third_class)

titanic_indep.Southampton = pd.CategoricalIndex(titanic_indep.Southampton)

titanic_indep.Cherbourg = pd.CategoricalIndex(titanic_indep.Cherbourg)



#Turn sex variable into a dummy

titanic_indep['Sex'] = titanic_indep.Sex.cat.codes
# Take the age column seperate and split them in the section with Na's and without Na's

agecolumn = titanic_indep["Age"]

Ages_noNA = agecolumn[agecolumn > -1]

Ages_yesNa = agecolumn[agecolumn.isna()]



# Normalize the section without Na's and add both sections back together

Ages_noNA = (Ages_noNA - Ages_noNA.mean()) / (Ages_noNA.max() - Ages_noNA.min())

agecolumn = Ages_noNA.append(Ages_yesNa, ignore_index=False)
# Take all columns except age

restcolumns = titanic_indep.loc[:, titanic_indep.columns != "Age"]

restcolumns = restcolumns.apply(pd.to_numeric)



# Apply normalization to each column of this dataframe

for i in (range(0, len(list(restcolumns)))):

    restcolumns.iloc[:,[i]] = ((restcolumns.iloc[:,[i]] - restcolumns.iloc[:,[i]].mean()) / 

                               (restcolumns.iloc[:,[i]].max() - restcolumns.iloc[:,[i]].min()))
#Add the normalized age columns to the dataframe

restcolumns["Age"] = agecolumn

titanic_indep = restcolumns
from fancyimpute import KNN



#We use the train dataframe from Titanic dataset

#fancy impute removes column names, so let's save them.

titanic_cols = list(titanic_indep)



# Use Knn to fill in each value and add the column names back to the dataframe

titanic_indep = pd.DataFrame(KNN(k = 9).fit_transform(titanic_indep))

titanic_indep.columns = titanic_cols

titanic_indep["Age"] = round(titanic_indep["Age"])
from sklearn import preprocessing

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(titanic_indep, titanic_dep, test_size=0.3)



#Tree packages for checking the feature importance

from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier() 



# Build a forest and compute the feature importances

## Fit the model on your training data.

rf.fit(X_train, y_train) 



rf_predictions = rf.predict(X_test)

## And score it on your testing data.

rf.score(X_test, y_test)
#perform logistic regression and KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



# Do a Kfold cross validation on the training data for k = 3



knn = KNeighborsClassifier(n_neighbors = 7)

CVscores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = "accuracy")

print("knn score:", CVscores.mean())



knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)



# Do a Kfold cross validation on the training data for a logistic regression



logregression = LogisticRegression(solver='liblinear')

CVscores = cross_val_score(logregression, X_train, y_train, cv = 10, scoring = "accuracy")

print("logistic regression score:", CVscores.mean())



        

logregression.fit(X_train, y_train)        

logreg_predictions = logregression.predict(X_test)

#Print the accuracy scores for each model.

from sklearn.metrics import accuracy_score



print("Random forest fit score",accuracy_score(rf_predictions, y_test))

print("Knn fit score", accuracy_score(knn_predictions, y_test))

print("Logistic regression fit score", accuracy_score(logreg_predictions, y_test))
#Let's see what the accuracy of the combined score would be:

combined_predictions = (rf_predictions + knn_predictions + logreg_predictions)/3

combined_predictions[combined_predictions < 0.5] = 0

combined_predictions[combined_predictions > 0.5] = 1

print("Combined predictions fit score", accuracy_score(combined_predictions, y_test))
Log_regression_model = logregression.fit(titanic_indep, titanic_dep)

import pickle



# save the model to disk

filename = 'Logistic_reg_model.sav'

pickle.dump(Log_regression_model, open(filename, 'wb'))