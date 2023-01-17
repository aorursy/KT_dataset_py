import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

train = pd.read_csv("../input/train.csv")

print("Number of Rows = " + str(len(train)))

print()

print("Number of Rows with missing values by column:")

pd.isnull(train).sum()
age_survived = train.loc[train['Survived']==1, "Age"]

age_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")

age_did_not_survive = train.loc[train['Survived']==0, "Age"]

age_did_not_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")

plt.xlabel("Age")

plt.ylabel("Percentage of Passengers")

plt.legend(loc='upper right')

plt.title("Distribution of Age of Survivors and Non-Survivors")
train = train[np.isfinite(train['Age'])]

print("Number of Rows = " + str(len(train)))
# Survived 

num_males = len(train.loc[train["Sex"]=="male",])

num_females = len(train.loc[train["Sex"]=="female",])



rates = train.loc[train["Survived"]==1, ["Pclass","Sex","PassengerId"]]

rates = pd.DataFrame(rates.groupby(["Pclass","Sex"]).count())

rates.reset_index(inplace=True)  

rates["Percentage"]=0



# adding a percentage column to show the percentage of males and females

for row in range(len(rates)):

    if rates.loc[row,"Sex"]=="male":

        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_males)*100,2)

    else:

        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_females)*100,2)



sns.set_style("whitegrid")

sns.barplot(x="Pclass", y="Percentage", hue="Sex", data=rates).set_title("Percentage of Survivors by Class")
# Did not Survive

num_males = len(train.loc[train["Sex"]=="male",])

num_females = len(train.loc[train["Sex"]=="female",])



rates = train.loc[train["Survived"]==0, ["Pclass","Sex","PassengerId"]]

rates = pd.DataFrame(rates.groupby(["Pclass","Sex"]).count())

rates.reset_index(inplace=True)  

rates["Percentage"]=0 



# adding a percentage column to show the percentage of males and females

for row in range(len(rates)):

    if rates.loc[row,"Sex"]=="male":

        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_males)*100,2)

    else:

        rates.loc[row, "Percentage"] = round((rates.loc[row,"PassengerId"]/num_females)*100,2)



sns.set_style("whitegrid")

sns.barplot(x="Pclass", y="Percentage", hue="Sex", data=rates).set_title("Percentage of Non-Survivors by Class")

num_passengers = len(train)

classes = pd.DataFrame(train.groupby(["Pclass"]).count())

classes.reset_index(inplace=True)

classes["Percentage"] = round(classes["PassengerId"].div(num_passengers)*100,2)



sns.set_style("darkgrid")

sns.pointplot(x="Pclass", y="Percentage", data=classes).set_title("Proportion of Passengers by Ticket Classes")
survivors_1 = train.loc[(train["Survived"]==1)&(train["Pclass"]==1), "Age"]

survivors_2 = train.loc[(train["Survived"]==1)&(train["Pclass"]==2), "Age"]

survivors_3 = train.loc[(train["Survived"]==1)&(train["Pclass"]==3), "Age"]



survivors_1.plot.hist(fc=(0, 0, 1, 0.5), label="Class 1")

survivors_2.plot.hist(fc=(1, 0, 0, 0.5), label="Class 2")

survivors_3.plot.hist(fc=(0, 1, 0, 0.5), label="Class 3")

plt.xlabel("Age")

plt.legend(loc='upper right')

plt.title("Age Distribution of Survivors by Ticket Class")
survivors_1 = train.loc[(train["Survived"]==0)&(train["Pclass"]==1), "Age"]

survivors_2 = train.loc[(train["Survived"]==0)&(train["Pclass"]==2), "Age"]

survivors_3 = train.loc[(train["Survived"]==0)&(train["Pclass"]==3), "Age"]



survivors_1.plot.hist(fc=(0, 0, 1, 0.5), label="Class 1")

survivors_2.plot.hist(fc=(1, 0, 0, 0.5), label="Class 2")

survivors_3.plot.hist(fc=(0, 1, 0, 0.5), label="Class 3")

plt.xlabel("Age")

plt.legend(loc='upper right')

plt.title("Age Distribution of Non-Survivors by Ticket Class")
first_class_survived = train.loc[(train["Pclass"]==1)&(train["Survived"]==1), "Age"]

first_class_didnt_survive = train.loc[(train["Pclass"]==1)&(train["Survived"]==0), "Age"]

first_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")

first_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")

plt.xlabel("Age")

plt.ylabel("Proportion of Class 1 Passengers")

plt.legend(loc='upper right')

plt.title("Distribution of Age of Passengers in First Class")
second_class_survived = train.loc[(train["Pclass"]==2)&(train["Survived"]==1), "Age"]

second_class_didnt_survive = train.loc[(train["Pclass"]==2)&(train["Survived"]==0), "Age"]

second_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")

second_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")

plt.xlabel("Age")

plt.ylabel("Proportion of Class 2 Passengers")

plt.legend(loc='upper right')

plt.title("Distribution of Age of Passengers in Second Class")
third_class_survived = train.loc[(train["Pclass"]==3)&(train["Survived"]==1), "Age"]

third_class_didnt_survive = train.loc[(train["Pclass"]==3)&(train["Survived"]==0), "Age"]

third_class_survived.plot.hist(fc=(0, 0, 1, 0.5), normed=1, label="Survived")

third_class_didnt_survive.plot.hist(fc=(1, 0, 0, 0.5), normed=1, label="Did not Survive")

plt.xlabel("Age")

plt.ylabel("Proportion of Class 3 Passengers")

plt.legend(loc='upper right')

plt.title("Distribution of Age of Passengers in Third Class")
train["nFamily"] = train["SibSp"] + train["Parch"]

train.head()
count = train.groupby(["Survived","nFamily"]).count()

count.reset_index(inplace=True)

count = count[["Survived", "nFamily", "PassengerId"]]



# get percentage

num_passengers = len(train)

count["Percentage"] = round(count["PassengerId"].div(num_passengers),2)



fig,ax = plt.subplots()



for i in range(2):

    ax.plot(count[count.Survived==i].nFamily, count[count.Survived==i].Percentage, label="Survived = "+ str(i))



ax.set_xlabel("Number of Family Members")

ax.set_ylabel("Proportion of Passengers")

ax.legend(loc='best')
# Converting "Sex" into a numerical column; 1 for male, 0 for female

train["Sex"] = train["Sex"].astype("category")

train["Gender"] = train["Sex"].cat.codes



train_y = train.Survived

predictor_cols = ['Age', 'Pclass', 'Gender', 'SibSp', 'Parch']

train_X = train[predictor_cols]



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)

classifier = LogisticRegression(random_state=0)

classifier.fit(train_X, train_y)
# Read the test data

test = pd.read_csv('../input/test.csv')

test["Sex"] = test["Sex"].astype("category")

test["Gender"] = test["Sex"].cat.codes



test['Initial']=0

for i in test:

    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',

                         'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',

                                          'Other','Other','Other','Mr','Mr','Mr'],inplace=True)



test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33

test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36

test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5

test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22

test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=46



test_X = test[predictor_cols]

test_X = scaler.transform(test_X)

predictions = classifier.predict(test_X)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

my_submission.to_csv('submission.csv', index=False)