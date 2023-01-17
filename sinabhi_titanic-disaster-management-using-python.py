# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



#Combining the train and test data so that feature transformation can be applied to both

data_combined = data_train.append(data_test)



#Let's take a peek at the data

data_combined.sample(3)
#Name



#Taking out title from Names

data_combined['Title'] = data_combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

title_dict = {

    "Mr": "Mr",

    "Miss": "Miss",

    "Mrs": "Mrs",

    "Master": "Master",

    "Dr": "Rare",

    "Rev": "Rare",

    "Col": "Rare",

    "Mlle": "Miss",

    "Major": "Rare",

    "Sir": "Rare",

    "Ms": "Miss",

    "the Countess": "Rare",

    "Lady": "Rare",

    "Jonkheer": "Rare",

    "Mme": "Mrs",

    "Don": "Rare",

    "Capt": "Rare",

    "Dona": "Rare"

}



data_combined['Title'] = data_combined['Title'].map(title_dict)

print(data_combined.groupby(['Sex','Title']).size())
# Family



# Instead of having two columns Parch & SibSp, 

#We can have the total family size for each individual

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

data_combined['FamilySize'] =  data_combined["Parch"] + data_combined["SibSp"] + 1

data_combined['Singleton'] = data_combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

data_combined['SmallFamily'] = data_combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)

data_combined['LargeFamily'] = data_combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)



# plot

sns.factorplot('FamilySize', hue='Survived', data=data_combined,kind='count')
# Embarked



# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

data_combined["Embarked"] = data_combined["Embarked"].fillna("S")



# plot

sns.factorplot('Embarked','Survived', data=data_combined,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=data_combined, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=data_combined, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = data_combined[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# Age - We are passing age as a numerical column



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# get average, std, and number of NaN values in data_train

average_age_titanic   = data_combined["Age"].mean()

std_age_titanic       = data_combined["Age"].std()

count_nan_age_titanic = data_combined["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)



# plot original Age values

# NOTE: drop all null values, and convert to int

data_combined['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

data_combined["Age"][np.isnan(data_combined["Age"])] = rand_1



# convert from float to int

data_combined['Age'] = data_combined['Age'].astype(int)

        

# plot new Age Values

data_combined['Age'].hist(bins=70, ax=axis2)
#Binning Age into a new feature 'AgeClass'



def getAgeClass(age):

    if (age<=16):

        return 0

    elif (age<=32):

        return 1

    elif (age<=48):

        return 2

    elif (age<=64):

        return 3

    else:

        return 4



data_combined['AgeClass'] = data_combined['Age'].apply(getAgeClass)



data_combined['AgeClass'].value_counts()
# Fare



# only for data_test, since there is a missing "Fare" values

data_combined["Fare"].fillna(data_combined["Fare"].median(), inplace=True)



# convert from float to int

data_combined['Fare'] = data_combined['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = data_combined["Fare"][data_combined["Survived"] == 0]

fare_survived     = data_combined["Fare"][data_combined["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

data_combined['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
# Cabin



data_combined['Cabin'].fillna('N',inplace=True)

data_combined['Cabin'].fillna('N',inplace=True)



data_combined['Cabin'] = data_combined.Cabin.apply(lambda c:c[0])
# Child and Mother



def get_mother(details):

    sex,parch,age,title = details

    return 1 if (sex=='female' and parch>0 and age>18 and title!='Miss') else 0



data_combined['isChild'] = data_combined['Age'].apply(lambda age: 1 if age<18 else 0)

data_combined['isMother'] = data_combined[['Sex','Parch','Age','Title']].apply(get_mother, axis=1)
# Ticket



def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        #ticket = map(lambda t : t.strip(), ticket)

        #ticket = filter(lambda t : not t.isdigit(), ticket)

        if len(ticket) > 1:

            return ticket[0].strip()

        else: 

            return 'XXX'

        return ticket

        

data_combined['Ticket'] = data_combined['Ticket'].apply(cleanTicket)
# drop unnecessary columns, these columns won't be useful in analysis and prediction

data_combined = data_combined.drop(['Name', 'Parch', 'SibSp', 'isChild', 'isMother', 'Singleton'], axis=1)



data_train = data_combined.head(891).drop(['PassengerId'], axis=1)

data_test    = data_combined.tail(418).drop(['Survived'], axis=1)
data_combined.head()
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Cabin','Embarked','Sex','Ticket','Title']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



data_train, data_test = encode_features(data_train, data_test)

data_train.head()
from sklearn.model_selection import train_test_split

X_all = data_train.drop('Survived', axis=1)

Y_all = data_train['Survived']



#We are keeping the ratio between training and validation set to be 0.20

test_ratio = 0.20

X_train,X_test,Y_train,Y_test = train_test_split(X_all, Y_all, test_size=test_ratio, random_state=23)
#Using XGBoost algorithm



clf = xgb.XGBClassifier()



#Choose some parameter combinations to try

parameters = {'n_estimators': [500,1000,1500], 

              'learning_rate': [0.02, 0.03, 0.04], 

              'max_depth': [1, 2, 3, 5]

              }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)
# Using SVM



clf = SVC()



parameters = {'kernel': ['rbf','linear'],

              'C': [1, 10, 20]

            }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)
#Using Random Forest Classifier



clf = RandomForestClassifier()



#Choose some parameter combinations to try

parameters = {'n_estimators': [10, 12, 15, 20], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [10, 12, 15], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)
# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)



print(accuracy_score(Y_test, predictions))
from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        Y_train, Y_test = Y_all.values[train_index], Y_all.values[test_index]

        clf.fit(X_train, Y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
ids = data_test['PassengerId']

predictions = clf.predict(data_test.drop('PassengerId', axis=1)).astype(int)





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()