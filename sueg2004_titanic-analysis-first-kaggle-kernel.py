# Import packages

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Open dataset

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

combined_data = [train_data, test_data]
# Look at first rows in training set

train_data.head()
# Look at first rows in test set

test_data.head()
# Get summary data for training set

train_data.describe()
# Now look at the test set

test_data.describe()
# Look at how many null values there are for the different columns

train_data.info()
train_data = train_data.drop(['Cabin'], axis='columns')

test_data  = test_data.drop(['Cabin'], axis='columns')



train_data.info()

print()

test_data.info()
# For simplicity, I will set the variabes with null values to the mean and 

# median of the non-null values.

train_data.Age  = train_data.Age.fillna(train_data.Age.median() )

train_data.Fare = train_data.Fare.fillna(train_data.Fare.mean() )



test_data.Age  = test_data.Age.fillna(test_data.Age.median() )

test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean() )
train_data.info()

print()

test_data.info()
# There are still 2 null values for the column Embarked.  Since we can't apply a mean/median 

# for this value, we will put in the most used value in the dataset

train_data.Embarked.value_counts()
# I will set Embarked = "S" where it is null

train_data.Embarked  = train_data.Embarked.fillna("S")
# Use Seaborn category plot 

gender_plot = sns.catplot(x="Sex", col="Survived", data=train_data, kind="count", 

              height=4, aspect=.75)

(gender_plot.set_axis_labels("", "Count")

            .set_xticklabels(["Men", "Women"]) 

            .set_titles("{col_name} {col_var}") )
# Look at distribution of survivors by Pclass and gender

class_plot = sns.catplot(x="Pclass", col="Survived", hue="Sex", data=train_data,  

             kind="count", height=4, aspect=.75)
# Look at distribution of survivors by Age, classified into 10 groups

age_plot = plt.hist(train_data[train_data.Survived == 1].Age, bins=10)
# Look at distribution of survivors by number of siblings or spouse

plcass_plot = plt.hist(train_data[train_data.Survived == 1].SibSp)
# Convert the gender to 0 (male) and (female)

train_data.Sex = [0 if i=="male" else 1 for i in train_data.Sex]

test_data.Sex  = [0 if i=="male" else 1 for i in test_data.Sex]



# Drop all of the columns that were not statistically significant

# (Cabin has already been removed)

new_train_x = train_data.drop(["PassengerId", "Survived", "Name", "Parch", "Ticket", 

                               "Fare", "Embarked"], axis=1)



new_train_y = train_data["Survived"]



# Drop the variables from the test data as well

new_test_x =  test_data.drop(["PassengerId", "Name", "Parch", "Ticket", 

                               "Fare", "Embarked"], axis=1)
# X is the training set, y is the prediction 

y = new_train_y 

X = new_train_x



# Split the training set in two, with 20% going to test set

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = .20, random_state = 0)
# Try Logistic Regression model

from sklearn.linear_model import LogisticRegression



reg = LogisticRegression()

reg.fit(train_x, train_y)

print("Accuracy - Logistic Regression - train:", round(reg.score(train_x, train_y), 3),

      "test:", round(reg.score(test_x, test_y), 3) )
# Try gradient boosting model

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score



GBC = GradientBoostingClassifier()

GBC.fit(train_x, train_y)



# Predicting the test set results

pred_y = GBC.predict(test_x)



# Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, pred_y)



print ("Accuracy - GradientBoosting - train:", round(GBC.score(train_x , train_y), 3),

       "test:", round(GBC.score(test_x , test_y), 3) )
# Try other algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier



# Random Forest

RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(train_x, train_y)

print ("Accuracy - RFC - train:", round(RFC.score(train_x , train_y), 3),

       "test:", round(RFC.score(test_x , test_y), 3) )
# Support Vector Classification

svc = SVC()

svc.fit(train_x, train_y)

print ("Accuracy - SVC - train:", round(svc.score(train_x , train_y), 3),

       "test:", round(svc.score(test_x , test_y), 3) )
# k-nearest neighbors

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(train_x, train_y)

print ("Accuracy - KNN - train:", round(knn.score(train_x , train_y), 3),

       "test:", round(knn.score(test_x , test_y), 3) )
# Create the submission file, with PassengerId and Survival prediction

psgr_id = test_data["PassengerId"]



# Rerun the model on the entire training set

GBC = GradientBoostingClassifier()

GBC.fit(new_train_x, new_train_y)



# Apply the algorithm to the test data file

prediction = GBC.predict(new_test_x)



# Save the results to a csv file

submission = pd.DataFrame( {"PassengerId" : psgr_id, "Survived": prediction} )

submission.to_csv("submission.csv", index=False)