#Data analysis
import pandas as pd
from pandas import Series,DataFrame

import numpy as np

#Graphics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
#Read files into the program
test = pd.read_csv("../input/test.csv", index_col='PassengerId')
train = pd.read_csv("../input/train.csv", index_col='PassengerId')
print ("Basic statistical description:")
train.describe()
train.info()
#Age
#Survived vs not survived by age
Age_graph = sns.FacetGrid(train, hue="Survived",aspect=3)
Age_graph.map(sns.kdeplot,'Age',shade= True)
Age_graph.set(xlim=(0, train['Age'].max()))
Age_graph.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
# Since there is a missing value in the "Fare" variable, I imputed using the median
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare']  = test['Fare'].astype(int)

# Fare for passengers that survived & didn't survive  
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived     = train["Fare"][train["Survived"] == 1]

# Average and std for survived and not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

# Histogram 'Fare'
train['Fare'].plot(kind='hist', figsize=(8,3),bins=100, xlim=(0,50))
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax = train.boxplot(column='Fare', by=['Embarked','Pclass'], ax=ax)
plt.axhline(y=80, color='green')
ax.set_title('', y=1.1)

train[train.Embarked.isnull()][['Fare', 'Pclass', 'Embarked']]
Emb = train.set_value(train.Embarked.isnull(), 'Embarked', 'C')
y = train['Survived']
del train['Survived']
train = pd.concat([train, test])
train.info()
#Drop variables that we will not included in the model: (6)Ticket.'
train = train.drop(train.columns[[6]], axis=1)
#fit_transform Encode labels with value between 0 and n_classes-1. 
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Pclass'] = LabelEncoder().fit_transform(train.Pclass)
train['Embarked'] = LabelEncoder().fit_transform(train.Embarked)
train['Cabin'] = train.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
train['Cabin'] = LabelEncoder().fit_transform(train.Cabin)
train[['Sex','Pclass', 'Cabin', 'Embarked']][0:3]
#Used to create new pd Series from Name data that extracts the greeting used for their name to be used 
#as a separate variable
def greeting_search(words):
    for word in words.split():
        if word[0].isupper() and word.endswith('.'): 
                     #name into an array of "words" 
                     #These are evaluate using the isupper() and endswith() methods in a for loop
            return word
# apply the greeting_search function to the 'Name' column
train['Greeting']=train.Name.apply(greeting_search)
train['Greeting'].value_counts()
#greetings that occur 8 or less times and classify them under the moniker 'Rare',
train['Greeting'] = train.groupby('Greeting')['Greeting'].transform(lambda x: 'Rare' if x.count() < 9 else x) 

#tranform the data and drop the 'Name' series since it's no longer needed.
del train['Name']  

train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)
#This will be accessed via the groupby method in Pandas:
train.groupby(['Greeting', 'Sex'])['Age'].median()
#set using Lambda x
train['Age'] = train.groupby(['Greeting', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median()))
train.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train['NorFare'] = pd.Series(scaler.fit_transform(train.Fare.reshape(-1,1)).reshape(-1), index=train.index)
train['NorFare'][0:10]
train['Family_Size'] = train.SibSp + train.Parch
train['Family_Size'][0:10]
Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)
Pclass.head(5)
Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)
Cabins = pd.get_dummies(train['Cabin'], prefix='Cabin', drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked', drop_first=True)
#Scale Continuous Data
train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()
train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()
#train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()
train.info()
train = train.drop(train.columns[[0,2,3,4,5,6,8,10]], axis=1)
#Varibles that I dropped: Pclass, Age, SisSp, Parch, Fare, Cabin, Greeting, Family_Size, 

#List of variables
#Pclass	Sex	Age SibSp Parch	Fare Cabin(6) Embarked Greeting(8) NorFare Family_Size(10)
#Family_scaled	Age_scaled	
train.info()
#Concat modified data to be used for analysis, set to X and y values
data = pd.concat([train, Greetings, Pclass, Cabins, Embarked], axis=1)
data.info()
#Split the data back into its original training and test sets
test = data.iloc[891:]
X = data[:891]
#Create cross - validation set 
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)
clf = LogisticRegression()
def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    score = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.C = Cs[score.index(max(score))]
    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))
    print('Accuracy: %g' % (max(score)))
find_C(X_val, y_val)
answer = pd.DataFrame(clf.predict(test), index=test.index, columns=['Survived'])
answer.to_csv('answer.csv')
coef = pd.DataFrame({'Variable': data.columns, 'Coefficient': clf.coef_[0]})
coef