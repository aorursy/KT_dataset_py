import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import tree, metrics

from sklearn.model_selection import train_test_split
dead = pd.read_csv('gender_submission.csv', sep=',')

train_data= pd.read_csv('train.csv', sep=',')

test_data= pd.read_csv('test.csv', sep=',')

test_merge= test_data.merge(dead, on = 'PassengerId')

complete_data = [train_data, test_merge]

result = pd.concat(complete_data, ignore_index = True)
#result.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))



# view changes

#result.info()
result.head()
result.info()

result.isnull().sum()
#replace nan values in AGE feature with mean of age values

#result['Age'].dropna()

#result['Age'].fillna(result['Age'].mean(), inplace=True)

#import numpy as np

#replacing nan values with the mean value

#result['Age'] = result['Age'].replace(np.nan, 30)

def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'
titles = sorted(set([x for x in result.Name.map(lambda x: get_title(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)

print()
def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:

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
result['Title'] = result['Name'].map(lambda x: get_title(x))
result['Title'] = result.apply(replace_titles, axis=1)
print('Title column values. Males and females are the same that for the "Sex" column:')

print(result.Title.value_counts())

result.head(20)
grouped = result.groupby(['Sex','Pclass', 'Title'])





# view the median Age by the grouped features

grouped.Age.median()
result.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))



# view changes

result.info()
result.loc[ result['Age'] <= 16, 'Age']= 0

result.loc[(result['Age'] > 16) & (result['Age'] <= 32), 'Age'] = 1

result.loc[(result['Age'] > 32) & (result['Age'] <= 48), 'Age'] = 2

result.loc[(result['Age'] > 48) & (result['Age'] <= 64), 'Age'] = 3

result.loc[ result['Age'] > 64, 'Age']                           = 4
result['Age'].head()
result.Cabin = result.Cabin.fillna('U')

#result['Cabin'] = result['Cabin'].replace(to_replace =['A','B','C','D','E','F','G','T',np.nan], value =[1,2,3,4,4,4,4,4,4])
result['Cabin'].head()
#result["Embarked"].fillna(lambda x: random.choice(result['Embarked'] != np.nan["Embarked"]), inplace =True)

result['Embarked'].fillna('S', inplace = True)
result['Embarked'].isnull().sum()
result['Embarked'] = result['Embarked'].replace(to_replace =['S','C','Q'], value =[1,2,3])
result['Embarked'].head()
result.isnull().sum()
result.Fare = result.Fare.fillna(result.Fare.median())

result['Fare'].head()

result.loc[result['Fare'] <= 10, 'Fare']= 0

result.loc[(result['Fare'] > 10) & (result['Fare'] <= 30), 'Fare'] = 1

result.loc[result['Fare'] > 30, 'Fare'] = 2



result.head()
result.Sex = result.Sex.map({"male": 0, "female":1})

result.head()
for index in range(0,len(result)):

    tuple=result['Cabin'].iloc[index]

    try:

        tuple = [c for c in result['Cabin'].iloc[index] if c.isalpha()]

        result['Fare'].iloc[index]=result['Fare'].iloc[index]/len(tuple)

        result['Cabin'].iloc[index]=tuple[0]

    except:

        a=1

result.head()
result['Cabin'] = result['Cabin'].replace(to_replace =['A','B','C','D','E','F','G','T','U'], value =[1,2,3,4,4,4,4,4,4])

result.head()
def bar_chart(feature):

    survived = result[result['Survived']==1][feature].value_counts()

    dead = result[result['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))

    plt.ylabel('Number of people')

    #plt.title('Plot to show how many female and male survived')
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
print("CORRELATION MATRIX")

corr = result.corr()

corr.style.background_gradient()
result.drop(columns = ['Name', 'Ticket'], inplace = True)
result.head()

result.info()
X = result.iloc[:,:-1]

y = result.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

dtree.fit(X_train, y_train)

# use the model to make predictions with the test data

y_pred = dtree.predict(X_test)

# how did our model perform?

#count_misclassified = (y_test != y_pred).sum()

#print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Decision Tree Accuracy: {:.2f}'.format(accuracy))
#from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

logreg = LogisticRegression()

print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))