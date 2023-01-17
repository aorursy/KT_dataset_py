# Import the required libraries

# Importing the basic dataFrame libraries

import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Import the visualization libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Import machine learning libraries

from sklearn.linear_model import LogisticRegression
# Import the train.csv file

train=pd.read_csv("../input/train.csv",

                  index_col='PassengerId')

test=pd.read_csv("../input/test.csv",

                 index_col='PassengerId')



# Converting Sex columns in 0 and 1 (Male =0, Female=1)

train["Sex"] = train["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)

test["Sex"] = test["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)
# Viewing the head of the train dataset

train.head()
# Describing the train data

train.describe()
# Correlation of the train dataset

sns.heatmap(train.corr(), annot = True)
# Information of the both train and test dataframe

train.info()

print('-----------------------------------')

test.info()



# Finding the null values in both the dataFrames

print('\n')

print(train.isnull().sum())

print('-----------------------------------')

print(test.isnull().sum())
# Counting the values of Survived

train.Survived.value_counts()



# Creating a pie chart of Passengers

Dead= (train['Survived']==0).sum()

Alive= (train['Survived']==1).sum()

proportion=[Dead, Alive]

plt.pie(proportion, labels=['Dead', 'Alive'], shadow=False,

       colors=['red', 'green'], explode=(0.1,0), startangle=45,

       autopct='%1.1f%%')

plt.axis('equal')

plt.title('Proportions of Survivors')

plt.tight_layout()

plt.show()
# Counting the nos. of Passengers Gender-wise

train.Sex.value_counts()



# Crosstab of no. of Survivors Gender-wise

cross=pd.crosstab(train.Survived, train.Sex)

print(cross)



# Plotting the stacked barchart of no. of Survivors Gender-wise

cross.plot.bar(stacked=True)

plt.xlabel('Survival Status')

plt.ylabel('No. of Passengers')

plt.title('No. of Survivors Gender-wise (Male =0, Female=1)')
# Counting the people survived from each Embarkment 

embark=pd.crosstab(train.Embarked, train.Survived)

# embark.rename(index=str, columns={"C": "Cherbourg", "Q": "Queenstown","S":"Southampton"})

print(embark)



# CrossTab for the Survivors from each each Embarkment

embark.plot.bar(stacked=True)

plt.xlabel('Port of Embarkment')

plt.ylabel('No. of Passengers')

plt.title('No. of Survivors Port-wise')
# Scatter Plot for age distribution of Passengers boarded

lm = sns.lmplot(x = 'Age', y = 'Survived', data = train, hue = 'Sex', fit_reg=False)



# set title

lm.set(title = 'Survived x Age')



# get the axes object and tweak it

axes = lm.axes

axes[0,0].set_ylim(-.1,)

axes[0,0].set_xlim(-5,85)
# Scatter Plot between the Fare paid and the Age

lm=sns.lmplot(x='Age', y='Fare', data=train, hue='Sex', fit_reg=False)

lm.set(title='Fare Vs Age')
# Practice barplot on seaborn 

bar=sns.barplot(x='Embarked',y='Survived', data=train, hue='Sex', orient='v')

bar.set(title='No.of Survivors Embarkment-wise')
# Boxplots of the fare payed

plt.figure(figsize=(10,7))

box=sns.boxplot(x='Survived', y='Fare', data=train, orient='v', saturation=0.75, width=0.8)

box.set(title="BoxPlot of Fare Paid by Passengers")
# Creating dummy Embarked column

train['Embarked']=train['Embarked'].fillna('S')

embark_dummies = pd.get_dummies(train.Embarked, prefix='Embark')

embark_dummies.drop(embark_dummies.columns[2], axis=1, inplace=True)

embark_dummies.sample(n=5, random_state=1)
# Creating the dummy class for Pclass

pclass_dummies=pd.get_dummies(train.Pclass, prefix='Class')

pclass_dummies.drop(pclass_dummies.columns[2], axis=1, inplace=True)

pclass_dummies.sample(n=5, random_state=1)
# Concatenating into one dataFrame "titanic_train"

titanic_train= pd.concat([train, embark_dummies, pclass_dummies], axis=1)

titanic_train.sample(n=5, random_state=1)



titanic_train.drop(['Embarked', 'Pclass', 'SibSp', 'Parch'], axis=1, inplace=True)

titanic_train.head()
# Calculating the no. of people in a family

titanic_train['Family_members']= train.SibSp + train.Parch + 1

titanic_train.head()
# Spliting the data into the Salutation

name = train.Name

def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'
# Finding the Salutation titles

titles = sorted(set([x for x in train.Name.map(lambda x: get_title(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)
# Replacing the titles

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

titanic_train['Title'] = train['Name'].map(lambda x: get_title(x))

titanic_train.Title.value_counts()
# Replacing the titles

titanic_train['Title'] = titanic_train.apply(replace_titles, axis=1)
# Counting the titles used

print('Title column values. Males and females are the same that for the "Sex" column:')

print(titanic_train.Title.value_counts())

titanic_train.Title.value_counts().plot(kind='bar')
# Encoding the Titles

# Importing the LabelEncode() from sklearn.preprocessing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

titanic_train['Title'] = le.fit_transform(titanic_train['Title'])
titanic_train.head()
# Creating the dummy class for Titiles

title_dummies=pd.get_dummies(titanic_train.Title, prefix= 'Title')

title_dummies.drop(title_dummies.columns[3], axis=1, inplace=True)

title_dummies.sample(n=5, random_state=1)



# Concatenating it to table titanic_train

titanic_train= pd.concat([titanic_train, title_dummies], axis=1)

titanic_train.sample(n=5, random_state=1)



titanic_train.drop(['Title'], axis=1, inplace=True)

titanic_train.head()
# Finding the correlation between the columns in titanic_train dataset

plt.figure(figsize = (12,10))

sns.heatmap(titanic_train.corr(), annot = True)
# Filling the missing age data with median

titanic_train["Age"] = titanic_train.groupby("Sex").transform(lambda x: x.fillna(x.median()))

print(titanic_train.info())

titanic_train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

print("\n")

print(titanic_train.info())
# Using the Train Test Split on the dataset

# Defing the X ad y

X = titanic_train.drop("Survived", axis=1)

y = titanic_train["Survived"]



# Importing the desired library

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30)
# Applying logistic regression model on training set

# Initializing the Logit function

logit=LogisticRegression(C=1e-2)



# Fitting the model

print(logit.fit(xTrain, yTrain))



#Predicting the model

print('\n')

print(logit.predict(xTest))    # Predicting the yTest value

print('\n')

print(logit.predict(xTrain))   # Predicting the yTrain value
# Finding the metric score

# Importing the required library

from sklearn import metrics

#print(metrics.accuracy_score(yTrain, logit.predict(xTrain)))

print(metrics.accuracy_score(yTest, logit.predict(xTest)))
# Filling NaN values in column in test with median group wise

test["Age"] = test.groupby("Sex").transform(lambda x: x.fillna(x.median()))

test["Fare"].fillna(test["Fare"].mean(), inplace=True)
# Making test set at par with training set

# Making dummy embarked columns

test['Embarked']=test['Embarked'].fillna('S')

embark_dummies_test = pd.get_dummies(test.Embarked, prefix='Embark')

embark_dummies_test.drop(embark_dummies_test.columns[2], axis=1, inplace=True)

embark_dummies.sample(n=5, random_state=1)



# Creating the dummy class for Pclass

pclass_dummies_test=pd.get_dummies(test.Pclass, prefix='Class')

pclass_dummies_test.drop(pclass_dummies_test.columns[2], axis=1, inplace=True)

pclass_dummies_test.sample(n=5, random_state=1)



# Concatenating into one dataFrame

titanic_test= pd.concat([test, embark_dummies_test,

                         pclass_dummies_test], axis=1)

titanic_test.sample(n=5, random_state=1)

titanic_test.drop(['Embarked', 'Pclass', 'SibSp', 'Parch'], axis=1, inplace=True)

titanic_test.head()



# Calculating the no. of people in a family

titanic_test['Family_members']= test.SibSp + test.Parch + 1



# Spliting the data into the Salutation

name = test.Name

def get_title_test(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



# Finding the Salutation titles

titles = sorted(set([x for x in test.Name.map(lambda x: get_title_test(x))]))

print('Different titles found on the test dataset:')

print(len(titles), ':', titles)



# Replacing the titles

def replace_titles_test(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    elif title == 'Dona':

        return 'Mrs'

    else:

        return title

titanic_test['Title'] = test['Name'].map(lambda x: get_title_test(x))



# Applying the titles

titanic_test['Title'] = titanic_test.apply(replace_titles_test, axis=1)



# Applying LabelEncoder to titanic_test

titanic_test['Title'] = le.fit_transform(titanic_test['Title'])



# Creating the dummy for title in test dataset 

title_dummies_test=pd.get_dummies(titanic_test.Title, prefix = 'Title')

title_dummies_test.drop(title_dummies_test.columns[3], axis = 1, inplace = True)

titanic_test = pd.concat([titanic_test, title_dummies_test], axis=1)

title_dummies_test.sample(n = 5, random_state = 1)

titanic_test.drop(['Title'], axis = 1, inplace = True)



# Filling the missing age data with median

titanic_test["Age"] = titanic_test.groupby("Sex").transform(lambda x: x.fillna(x.median()))

print(titanic_test.info())

titanic_test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

print("\n")

print(titanic_test.info())
# Predicting the test model

#titanic_test = titanic_test.drop('Title_4', axis = 1)

y_pred_1 = logit.predict(titanic_test)



# Printing the predictions

log_model_1 = pd.DataFrame({'PassengerId' : np.arange(892,1310), 'Survived': y_pred_1})

log_model_1.head()
# Printing the predicted value found in the test dataset

print(y_pred_1)
# Submitting the model

log_model_1.to_csv('Titanic_Submission_.csv', index=False)