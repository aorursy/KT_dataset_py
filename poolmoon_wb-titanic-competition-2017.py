import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# We can use the pandas library in python to read in the csv file.

# This creates a pandas dataframe and assigns it to the titanic variable.

titanic = pd.read_csv("../input/train.csv")



#Check out the data

titanic.describe()

titanic.head()
# Let's Clean up some data.



# Replace na data for Age and Fare

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



# Replace all the occurences of male with the number 0 and female with number 1.

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



# Numerica values for Embarked

# print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic_test = pd.read_csv("../input/test.csv")



titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())



titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1



titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
# Use only these features to make preditions

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# TODO: Split your training data. It might be a good idea to use K-fold. 
# TODO: ID3 decision trees

# Hint: You can write your own code, or use ID3 classifier from sklearn 
# TODO: CART decision trees

# Hint: You can write your own code, or use CART classifier from sklearn 
# TODO: Use information gain ratio as your criteria

# Hint: there is not existing realization in the sklearn lib. Your may want to add your own.
# TODO: Can you create an illustration of the trees?
# TODO: Use accuracy score to evaluate your results (the competition criteria)
# TODO: !!!Replace the following line with your own classifier!!!



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)



# Train the algorithm using all the training data

clf.fit(titanic[predictors], titanic["Survived"])



# Make predictions using the test set.

predictions = clf.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("kaggle.csv", index=False)