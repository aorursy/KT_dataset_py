import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
training_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



#Let's take a look at the first 10 rows from the training dataset

training_df.head(10)

training_df.dtypes ##Examine data types of each column
training_df.describe() #get some summary statistics on the numerical variables
training_df.isnull().sum() ##get sum of null/NaN rows



#There are 177 null or nan rows in the Age column as are a whopping 687 (about 80%) missing rows in the Cabin column. 

#Embarked has only 2 missing or NaN rows
test_df.isnull().sum() ##get sum of null/NaN rows for test dataset
#Plot histogram on the Age column to show the distribution

training_df.Age.hist()



#The histogram tells that there are more youth (20 - 35 years) in the dataset
#The distribution of Fares

training_df.Fare.hist()
#A bar plot showing the survival ratio by the category of ticket (pClass)

training_df.groupby('Pclass').mean()["Survived"].plot(kind='bar')



#It appears that passengers with class A or 1st class tickets had higher chances of surving that passengers with 3rd ticket class
#Bar plot showing the survival ration by gender or Sex. We can see more female survived than did male

training_df.groupby('Sex').mean()[["Survived"]].plot(kind='bar')
sns.violinplot(x='Sex', y='Age', hue='Survived', data=training_df, split=True, scale="count", inner="quartile") 

#Of the females who survived, the violinplot below shows that a greater proportion of them (as shown by the distribution) were

# within the age range of 25 to 30 years
sns.violinplot(x='Embarked', y='Age', hue='Survived', data=training_df, split=True, scale="count", inner="quartile") ##distribution of embarked with age



##It is also clear from the plot below that passengers who board the ship at C had higher chance of surviving, followed by S and the Q
#Let's also plot a correlation matric to investigate which features correlate with others.

corr_matrix = training_df.corr()

corr_matrix
sns.heatmap(corr_matrix) ##correlation matric to visualize the relationship with the target variable
##inpute age with mean since it has missing data

training_df.Age.fillna(training_df.Age.mean(), inplace=True)

test_df.Age.fillna(test_df.Age.mean(), inplace=True)

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)
 ##drop the Cabin column as it has over 600 missing values

training_df.drop('Cabin', axis=1, inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)
training_df.info()

test_df.info()
#Process family

def process_family(dataset):

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['SoloFamily'] = dataset.FamilySize.map(lambda x: 1 if x==1 else 0)

    dataset['SmallFamily'] = dataset.FamilySize.map(lambda x: 1 if 2 <= x <=4 else 0)

    dataset['LargeFamily'] = dataset.FamilySize.map(lambda x: 1 if x >= 5 else 0)

    dataset.drop(['SibSp','Parch'], axis=1, inplace=True)

    return dataset
training_df = process_family(training_df)



test_df = process_family(test_df)

training_df.head()
#process gender

def process_gender(data):

    gender_map = {'male':0, 'female':1}

    data['Sex'] = data.Sex.map(lambda x: 1 if x=='male' else 0)

    return data
training_df = process_gender(training_df)

test_df = process_gender(test_df)

training_df.head()
#process embarked

def process_embarked(data):

    data.Embarked.fillna(data['Embarked'].mode(), inplace=True)

    #one hot encode process embarked

    encoded_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')

    data = pd.concat([data, encoded_embarked], axis=1)

    data.drop(["Embarked"], axis=1, inplace=True)

    return data

    
training_df = process_embarked(training_df)



test_df = process_embarked(test_df)

training_df.head()

##Generate unique titles from dataset
def get_titles(data):

    titles = set()

    for title in data:

        titles.add(title.split(",")[1].split(".")[0].strip())

    return titles

titles = get_titles(training_df.Name)

titles
#process titles

##Portion of code referenced from https://towardsdatascience.com/kaggle-titanic-machine-learning-model-top-7-fa4523b7c40

def process_names(data):

    title_dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

    }

    data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    data['Title'] = data['Title'].map(title_dictionary)

    data.drop(['Name'], axis=1, inplace=True)

    #one hot encode titles

    titles_dummies = pd.get_dummies(data['Title'], prefix="Title")

    data = pd.concat([data, titles_dummies], axis=1)

    data.drop(['Title'], axis=1, inplace=True)



    return data
training_df = process_names(training_df)



test_df = process_names(test_df)

training_df.head()

test_df.head(5)
##drop ticket

training_df.drop(['Ticket'], axis=1, inplace=True)

test_df.drop(['Ticket'], axis=1, inplace=True)



training_df.head()
##Scale numerical variables

scaler = StandardScaler()

training_df[['Age','Fare']] = scaler.fit_transform(training_df[['Age','Fare']])

training_df.head(10)



test_df[['Age','Fare']] = scaler.fit_transform(test_df[['Age','Fare']])

X_Train = training_df.drop(['Survived','PassengerId'], axis=1)

Y_Train =training_df.iloc[:,1]

X_Test = test_df

X_Train.head()
#Import sklearn models

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
#X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state=0)



r_forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features =0.5)

r_forest.fit(X_Train, Y_Train)

Y_Predicted = r_forest.predict(X_Test)

feature_importances = pd.DataFrame(r_forest.feature_importances_,

                                   index = X_Train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
#X_Train.drop(['Embarked_C','Title_Royalty','SoloFamily','Embarked_Q',], axis=1, inplace=True)

#X_Test.drop(['Embarked_C','Title_Royalty','SoloFamily','Embarked_Q',], axis=1, inplace=True)

#X.drop('PassengerId', axis=1, inplace=True) ##This is not needed

#X_Train.head()


logistic_model = LogisticRegression(random_state=0)

logistic_model.fit(X_Train, Y_Train)

model_score = logistic_model.score(X_Train, Y_Train)

parameters = logistic_model.coef_

print(parameters)

print(model_score)
#Our Logistic Regression model scored 82.8%
naive_model = GaussianNB()

naive_model.fit(X_Train, Y_Train)

Y_Predict = naive_model.predict(X_Test)

naive_model.score(X_Train, Y_Train)
##Fit a Support Vector Classifier
support_vector = SVC()

support_vector.fit(X_Train, Y_Train)

support_vector.score(X_Train, Y_Train)
##Gradient Boosting Descent Model
graident_boosting = GradientBoostingClassifier()

graident_boosting.fit(X_Train, Y_Train)

graident_boosting.score(X_Train, Y_Train)
#Of all the models tested, it appears the gradient boosting classifier outperformed the others with a score of 89.7%. 

#This explains why we're using it here

passengerId = X_Test["PassengerId"]

New_Test = X_Test.drop('PassengerId', axis=1)

Y_Predicted = graident_boosting.predict(X_Test)
#Create a pandas dataframe and convert it to a csv format (see instructions on how to upload your solution)

submission_dataFrame = pd.DataFrame({'PassengerId': passengerId, 'Survived': Y_Predicted})

submission_dataFrame.to_csv("submission.csv",index=False)