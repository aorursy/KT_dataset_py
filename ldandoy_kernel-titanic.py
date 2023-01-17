# First understand the data



import pandas as pd



df = pd.read_csv("/kaggle/input/titanic/train.csv") # Read the CSV file

print(df.count()) # Show the count of not null rows by columns

#df.head()
# Statistics



import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("/kaggle/input/titanic/train.csv")

fig = plt.figure(figsize=(18, 6))



plt.subplot2grid((2,3), (0, 0))

df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Survived")



plt.subplot2grid((2,3), (0, 1))

plt.scatter(df.Survived, df.Age, alpha=0.1)

plt.title("Age wrt Survived")



plt.subplot2grid((2,3), (0, 2))

plt.scatter(df.Survived, df.Parch, alpha=0.1)

plt.title("Parch wrt Survived")



plt.show() # Show the plot
import pandas as pd



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
# Clean the data (replace N/A value, convert string to integer)

def clean_data(data):

    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())

    data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

    

    data.loc[data['Sex'] == "male", "Sex"] = 0

    data.loc[data['Sex'] == "female", "Sex"] = 1

    

    data["Embarked"] = data["Embarked"].fillna("S")

    data.loc[data['Embarked'] == "S", "Embarked"] = 0

    data.loc[data['Embarked'] == "C", "Embarked"] = 1

    data.loc[data['Embarked'] == "Q", "Embarked"] = 2
import pandas as pd

from sklearn import linear_model

from sklearn.model_selection import train_test_split



# Load the data set

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# Make some cleaning

clean_data(train_data)



# Make some split on the data set

target = train_data['Survived'].values

features = train_data[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

target_train, target_test, features_train, features_test = train_test_split(target, features, test_size=0.1)



# For understanding => printing of the result

# print(target_train)

# print(target_test)

# print(features_train)

# print(features_test)



# Apply our linear_model on the data split

classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)



classifier2_ = classifier.fit(features_train, target_train)



# See the score on the trainning data we have on it

print('Difference between all the dataset and the split dataset:')

print("Full data: ", classifier_.score(features, target))

print("Split data", classifier2_.score(features_train, target_train))



# See the score on the test data we have on it

print('Difference between the 2 models after trainning on the same data set test:')

print("Full data: ", classifier_.score(features_test, target_test))

print("Split data", classifier2_.score(features_test, target_test))



# Make some split on the data set

target2 = train_data['Survived'].values

features2 = train_data[["Pclass", "Age", "Fare", "Embarked"]].values

classifier = linear_model.LogisticRegression()

classifier2_ = classifier.fit(features2, target2)

print('Without 3 colomn:')

print(classifier2_.score(features2, target2))



# Make some split on the data set

target3 = train_data['Survived'].values

features3 = train_data[["Pclass", "Fare", "Embarked"]].values

classifier = linear_model.LogisticRegression()

classifier3_ = classifier.fit(features3, target3)

print('Without 4 colomn:')

print(classifier3_.score(features3, target3))
import pandas as pd

from sklearn import linear_model

from sklearn.model_selection import train_test_split



# Load the data set

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

# Make some cleaning

clean_data(train_data)



# Make some split on the data set

target = train_data['Survived'].values

features = train_data[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

target_train, target_test, features_train, features_test = train_test_split(target, features, test_size=0.1)



# For understanding => printing of the result

# print(target_train)

# print(target_test)

# print(features_train)

# print(features_test)



# Apply our linear_model on the data split

classifier = linear_model.LogisticRegression()

classifier_ = classifier.fit(features, target)



classifier2_ = classifier.fit(features_train, target_train)



# See the score on the trainning data we have on it

print('Difference between all the dataset and the split dataset:')

print("Full data: ", classifier_.score(features, target))

print("Split data", classifier2_.score(features_train, target_train))



# See the score on the test data we have on it

print('Difference between the 2 models after trainning on the same data set test:')

print("Full data: ", classifier_.score(features_test, target_test))

print("Split data", classifier2_.score(features_test, target_test))



# Make some split on the data set

target2 = train_data['Survived'].values

features2 = train_data[["Pclass", "Age", "Fare", "Embarked"]].values

classifier = linear_model.LogisticRegression()

classifier2_ = classifier.fit(features2, target2)

print('Without 3 colomn:')

print(classifier2_.score(features2, target2))



# Make some split on the data set

target3 = train_data['Survived'].values

features3 = train_data[["Pclass", "Fare", "Embarked"]].values

classifier = linear_model.LogisticRegression()

classifier3_ = classifier.fit(features3, target3)

print('Without 4 colomn:')

print(classifier3_.score(features3, target3))