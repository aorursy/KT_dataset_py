# Import Libraries 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import ensemble

import seaborn as sns

# Ensures graphs to be displayed in ipynb

%matplotlib inline   
# read data into dataframe

titanic_df = pd.read_csv('../input/train.csv',header=0)  # Always use header=0 to read header of csv files

titanic_df.head()

#titanic_df.Age.median
# we can see that the passenger details have been imported and we have all kind of dataformats available for each data field. 

# Next step is to munging the data 

# lets describe and get the info of the data to do so

titanic_df.info()

# we can observe that couple of information for age,embarked and cabin are missing. Out of which Embarked and Age seems relevent
# Lets fill up missing values for age

#  assign to a new varaible before that so the data is not lost

cleaned_df = titanic_df

cleaned_df['Age'] = cleaned_df['Age'].fillna(cleaned_df.Age.median())

# cleaned_df.head()

cleaned_df[cleaned_df['Age'] > 60][['Age','Parch', 'Survived']].describe()
# we can see that passengers above 60 only 22% of people survived from the crash. 

#Just exploring how many people are there for each class

for i in range(1,4):

    print (i, ' male ' , len(cleaned_df[ (cleaned_df['Sex'] == 'male') & (cleaned_df['Pclass'] == i) ]))

    print (i, 'female' , len(cleaned_df[ (cleaned_df['Sex'] == 'female') & (cleaned_df['Pclass'] == i) ]))
# Lets also fix the embarked 

for i in ['S','C','Q']:

    print (i, len(cleaned_df[cleaned_df['Embarked'] == i]))
# we can see that we have maximum 'S' so let fill in with 'S' for those missing 2 values 

cleaned_df['Embarked'] = cleaned_df['Embarked'].fillna('S')
# Lets work on gender now , lets see the proportion of males survived vs female

total_male = len(cleaned_df[(cleaned_df['Sex'] == 'male')])

total_female = len(cleaned_df[(cleaned_df['Sex'] == 'female')])

num_males_survived = len(cleaned_df[(cleaned_df['Sex'] == 'male') & cleaned_df['Survived'] == 1])

num_females_survived = len(cleaned_df[(cleaned_df['Sex'] != 'male') & cleaned_df['Survived'] == 1])

print (num_males_survived/float(total_male) * 100 ,'% of males survived')

print (num_females_survived/float(total_female) * 100 ,'% of females survived')
# as it is hard to work on string data in ML lets convert the 'sex' to 'gender' and have values 0,1 for m and f

cleaned_df['Gender'] = cleaned_df['Sex'].map({'female':0, 'male':1}).astype(int)
# now lets cleanup the parch (parent and children) and siblings 

cleaned_df['Family'] = cleaned_df['Parch'] + cleaned_df['SibSp']
# Lets display all datatypes that are not good for machine learning, like string/objects

cleaned_df.dtypes[cleaned_df.dtypes.map(lambda x: x== 'object')]
# As they dont add any value we can drop them to create our train_data (training data)

train_data = cleaned_df.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)

# we can also drop SibSp,Parch as they are part of Family now

train_data = train_data.drop(['SibSp','Parch'],axis=1)
# Let us also prepare the test data in similar format as we did for train_data

titanic_test_df = pd.read_csv('../input/test.csv',header=0)

titanic_test_df.info()
# we do have 418 valies of which only 332 values are available for age , 

# we can also merge SibSp and Parch and change the Sex to gender etc.. as we did for our train data

# infact let us create a function to reuse the same for any kind of data

def clean_up_df(df):

    """ This function will cleanup Age(Median), Sex(Change to 0,1), SibSp,Parch(Merge to Family), Embarked data

    Update to 'S' And Also deletes Name,Cabin details from titanic DF, Ensure to Pass DataFrame to this Function"""

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

    df['Family'] = df['Parch'] + df['SibSp']

    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    df = df.drop(['SibSp','Parch','Sex','Name','Cabin','Embarked','Ticket'],axis=1)

    return df

# gender 
test_df = clean_up_df(titanic_test_df)

test_df.info()
# lets explore full data with sns pairplots

sns.pairplot(train_data,hue='Survived',size=3.5)
# Logistic Regression 

logistic = linear_model.LogisticRegression()

X = train_data.drop(['PassengerId','Survived'],axis=1)

y = train_data['Survived']

logistic.fit(X,y)

logistic.score(X, y)
X_test = test_df.drop(['PassengerId'],axis=1)
X_test.info()
y_pred = logistic.predict(X_test)
# Random Forests 

random_forest = ensemble.RandomForestClassifier(n_estimators=100)

random_forest.fit(X,y)

y_pred = random_forest.predict(X_test)

random_forest.score(X,y)
submission = pd.DataFrame({

        "PassengerId":test_df['PassengerId'],

        "Survived":y_pred

    })

submission.to_csv('titanic1.csv',index=False)
## Now that we completed our Basic Model we need to plot the learning curves which will help in improving the model