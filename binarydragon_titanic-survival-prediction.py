import numpy as np 

import pandas as pd



full_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

full_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.info()

print("\n\n --- \n\n")

test_data.info()
train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
del train_data["Cabin"]

del test_data["Cabin"]



print("Cabin columns deleted")
# Calculate the average age

print(" - Calculating average ages - \n\n")

average_train_age = 0

average_test_age = 0

count = 0

for age in train_data["Age"]:

    if not pd.isnull(age):

        count += age

average_train_age = count/train_data["Age"].count()



count = 0



for age in test_data["Age"]:

    if not pd.isnull(age):

        count += age

average_test_age = count/test_data["Age"].count()

print("Average training-set age: %s" % average_train_age)

print("Average test-set age: %s" % average_test_age)



print("\n\n - Rounding up - \n\n")



# Rounding up (to fit the format of the data, one decimal)...

average_train_age = round(average_train_age, 1)

average_test_age = round(average_test_age, 1)



print("Average training-set age: %s" % average_train_age)

print("Average test-set age: %s" % average_test_age)



# Assign it to the missing values



print("\n\n - Assigning average values - \n\n")

for i in range(0, train_data["PassengerId"].count()):

    if pd.isnull(train_data.iloc[i]["Age"]):

        train_data.at[i, "Age"] = average_train_age

for i in range(0, test_data["PassengerId"].count()):

    if pd.isnull(test_data.iloc[i]["Age"]):

        test_data.at[i, "Age"] = average_test_age

print("Done")
print("Embarked unique values count: %s" % train_data["Embarked"].nunique())



# Next, let's try list those values...

embarked_unique_vals = []



for loc in train_data["Embarked"]:

    if not loc in embarked_unique_vals:

       embarked_unique_vals.append(loc) 

print("Embarked unique values: %s" % embarked_unique_vals)



# This fits exactly the prediction we had earlier!
print("Training data: \n%s\n\n" % train_data["Embarked"].value_counts())
print("Before fix, number of null values: %s" % train_data["Embarked"].isnull().sum())

train_data["Embarked"].fillna("S", inplace=True)  

print("After fix, number of null values: %s" % train_data["Embarked"].isnull().sum())
fare_missing_value_row = 99999

for i in range(0, test_data["PassengerId"].count()):

    if pd.isnull(test_data.at[i, "Fare"]):

        fare_missing_value_row = i

print("Person: \n%s" % test_data.loc[fare_missing_value_row])

print("\n\n - Determining class - \n\n")

print("Class: %s" % test_data.at[fare_missing_value_row, "Pclass"])
test_fare_average = 0 

for i in range(0, test_data["PassengerId"].count()):

    if not pd.isnull(test_data.at[i, "Fare"]) and test_data.at[i, "Pclass"] == 3:

        test_fare_average += test_data.at[i, "Fare"]

test_fare_average /= test_data["PassengerId"].count()

print("Unrounded fare average: %s" % test_fare_average)

# Round to 4 decimals to fit the data's format

test_fare_average = round(test_fare_average, 4)

print("Rounded fare average: %s" % test_fare_average)

# Let's assign it now...

test_data.at[fare_missing_value_row, "Fare"] = test_fare_average

print("After fix, number of missing values in Fare (test data): %s" % test_data["Fare"].isnull().sum())

train_data.isnull().sum()
test_data.isnull().sum()
train_data.info()
test_data.info()
# Before the example, remove name as a column.

del train_data["Name"]

del test_data["Name"]



# So, how important is the gender feature? How many people from each gender survived (at least in the training data)? Let's see.

female_passengers = [train_data.iloc[i] for i in range(0, len(train_data.index)) if train_data.iloc[i][3] == "female"]

male_passengers = [train_data.iloc[i] for i in range(0, len(train_data.index)) if train_data.iloc[i][3] == "male"]



female_survivors = [female_passengers[i] for i in range(0, len(female_passengers)) if female_passengers[i][1]==1]

male_survivors = [male_passengers[i] for i in range(0, len(male_passengers)) if male_passengers[i][1]==1]



print("Number of male survivors (from training data): %s" % len(male_survivors))

print("Number of female survivors (from training data): %s" % len(female_survivors))



# Let's remember the total number of people in the training set.



print("Total number of passengers on the Titanic (from the training data): %s" % len(train_data.index))

print("Total number of female passengers on the Titanic (from the training data): %s" % len(female_passengers))

print("Total number of male passengers on the Titanic (from the training data): %s" % len(male_passengers))



# Thus... survival rate. (Rounded to 1 decimal)

print("Rounded female survival rate: %s%%" % (100 * round(len(female_survivors) / len(female_passengers), 3)))

print("Rounded male survival rate: %s%%" % (100 * round(len(male_survivors) / len(male_passengers), 3)))
# Let's drop Ticket first

del train_data["Ticket"]

del test_data["Ticket"]



# Now let's label encode sex

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



train_data["Sex"] = label_encoder.fit_transform(train_data["Sex"])

test_data["Sex"] = label_encoder.transform(test_data["Sex"])



# Let's take a look at how important "embarked" is

S_passengers = [train_data.iloc[i] for i in range(0, len(train_data.index)) if train_data.iloc[i][8] == "S"]

C_passengers = [train_data.iloc[i] for i in range(0, len(train_data.index)) if train_data.iloc[i][8] == "C"]

Q_passengers = [train_data.iloc[i] for i in range(0, len(train_data.index)) if train_data.iloc[i][8] == "Q"]



S_survivors = [S_passengers[i] for i in range(0, len(S_passengers)) if S_passengers[i][1]==1]

C_survivors = [C_passengers[i] for i in range(0, len(C_passengers)) if C_passengers[i][1]==1]

Q_survivors = [Q_passengers[i] for i in range(0, len(Q_passengers)) if Q_passengers[i][1]==1]



print("Number of Southampton survivors (from training data): %s" % len(S_survivors))

print("Number of Cherbourg survivors (from training data): %s" % len(C_survivors))

print("Number of Queenstown survivors (from training data): %s" % len(Q_survivors))



# Let's remember the total number of people in the training set.



print("Total number of passengers on the Titanic (from the training data): %s" % len(train_data.index))

print("Total number of Southampton passengers on the Titanic (from the training data): %s" % len(S_passengers))

print("Total number of Cherbourg passengers on the Titanic (from the training data): %s" % len(C_passengers))

print("Total number of Queenstown passengers on the Titanic (from the training data): %s" % len(Q_passengers))



# Thus... survival rate. (Rounded to 1 decimal)

print("Rounded Southampton survival rate: %s%%" % (100 * round(len(S_survivors) / len(S_passengers), 3)))

print("Rounded Cherbourg survival rate: %s%%" % (100 * round(len(C_survivors) / len(C_passengers), 3)))

print("Rounded Queenstown survival rate: %s%%" % (100 * round(len(Q_survivors) / len(Q_passengers), 3)))
# One-hot encode the "Embarked" value

from sklearn.preprocessing import OneHotEncoder



one_hot_enc = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse=False) # ignore values that aren't present in the training data, and return values as a numpy array. Though the first parameter is unlikely to be needed.



OH_cols_train = pd.DataFrame(one_hot_enc.fit_transform(train_data.Embarked.values.reshape(-1, 1))) 

OH_cols_test = pd.DataFrame(one_hot_enc.transform(test_data.Embarked.values.reshape(-1, 1)))



OH_cols_train.index = train_data.index

OH_cols_test.index = test_data.index



del train_data["Embarked"]

del test_data["Embarked"]



train_data = pd.concat([train_data, OH_cols_train], axis=1)

test_data = pd.concat([test_data, OH_cols_test], axis=1)

del train_data["PassengerId"]

del test_data["PassengerId"]
train_data.head()
test_data.head()
# Seperate the target, survived, from our training data.



y = train_data["Survived"]

train_data = train_data.drop(['Survived'], axis=1)



# Import XGRegressor

from xgboost import XGBRegressor



# Create model and fit

model = XGBRegressor()

model.fit(pd.get_dummies(train_data), y)



# Get predictions

predictions = model.predict(pd.get_dummies(test_data))



# Round each prediction up or down to get a whole 0, or 1.

predictions = [round(x) for x in predictions]

int_predictions = [int(x) for x in predictions]



# Save and submit

output = pd.DataFrame({'PassengerId': full_test_data.PassengerId, 'Survived': int_predictions})

output.to_csv('my_submission.csv', index=False)
output.head() #  Output first few predictions