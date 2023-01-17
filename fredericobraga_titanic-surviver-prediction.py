%matplotlib inline



#get data

import pandas as pd

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import re



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



aged_changed = False



warnings.filterwarnings('ignore')



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()

test_df.head()
#clean



survive_count = train_df.groupby('Survived').size() 



#erase useless columns for the prediction

traind_df = train_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)



print (train_df.describe())



print (train_df.head())
#pclass



pclass_count  = train_df.groupby('Pclass').size()

unique_pclass   = train_df['Pclass'].unique()



print('Pclass count',pclass_count )

print ('Pclass', unique_pclass)



sns.factorplot('Pclass', 'Survived',data=train_df, order=unique_pclass.sort())
#sex

sex_count     = train_df.groupby('Sex').size()

unique_sex      = train_df['Sex'].unique()



print( 'Sex count', sex_count)

print ('Sex', unique_sex)

sns.factorplot('Sex', 'Survived',data=train_df, order=unique_sex.sort())



train_df['Sex'] = train_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
#siblings and spouses



sibsp_count   = train_df.groupby('SibSp').size()

unique_sibsp    = train_df['SibSp'].unique()



print( 'SibSp count', sibsp_count)  

print ('SibSp', unique_sibsp)

#sns.factorplot('SibSp', 'Survived',data=train_df, order=unique_sibsp.sort())
#parents and children



parch_count  = train_df.groupby('Parch').size()

unique_parch = train_df['Parch'].unique()



print( 'Parch count', parch_count )

print ('Parch', unique_parch)

#sns.factorplot('Parch', 'Survived',data=train_df, order=unique_parch.sort())
# is Alone





train_df['isAlone'] = (train_df['SibSp'] + train_df['Parch']) == 0

train_df['isAlone']



is_alone_count  = train_df.groupby('isAlone').size()

unique_is_alone = train_df['isAlone'].unique()



print( 'isAlone count', is_alone_count )

print ('isAlone', unique_is_alone)

sns.factorplot('isAlone', 'Survived',data=train_df, order=unique_is_alone.sort())



traind_df = train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)



train_df.head()
# age



def split_equal_groups(age):

   

    if age < 23:

        return 0

    elif age < 33 or age == None:

        return 1

    elif age < 55:

        return 2

    elif age:

        return 3

    

if aged_changed == False :

    train_df['Age'] = train_df['Age'].apply(split_equal_groups)    

    aged_changed = True

    

#train_df['Age'].fillna(age_count.argmax(), inplace=True)



age_count  = train_df.groupby('Age').size()

unique_age      = train_df['Age'].unique()



print( 'Age count', age_count )   

print ('Age', unique_age)

plt.hist(train_df['Age'])

sns.factorplot('Age', 'Survived',data=train_df, order=unique_age.sort())



train_df['Age'].fillna(age_count.argmax(), inplace=True)
#name

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

    

def filter_title(title):

    

    if train_df.groupby('Title').size()[title] >= 40:

        return title

    else:

        return 'Other'

    

def filter_title_test(title):



    if test_df.groupby('Title').size()[title] >= 40:

        return title

    else:

        return 'Other'



def classify_title(title):

    

    if title == "Mr":

        return 0

    elif title == "Mrs":

        return 1

    elif title == "Miss":

        return 2

    elif title == "Master":

        return 3

    else: 

        return 4

    

train_df['Title'] = train_df['Name'].apply(get_title)

train_df['Name_Len'] = train_df['Name'].apply(lambda x: len(x))

train_df.drop(['Name'], axis=1, inplace=True)



train_df['Title'] = train_df['Title'].apply(filter_title)



title_count  = train_df.groupby('Title').size()

unique_title = train_df['Title'].unique()



print( 'Title count', title_count )   

print ('Title', unique_title)



train_df['Title'] = train_df['Title'].apply(classify_title)



train_df['Title']





    



#fare



train_df['Fare'].fillna(train_df['Fare'].mean())

#Embarked

def convert_embarked(embarked):

    if embarked == 'S':

        return 0

    elif embarked == 'C':

        return 1

    elif embarked == 'Q':

        return 2

    else:

        return 0





embarked_count  = train_df.groupby('Embarked').size()

unique_embarked = train_df['Embarked'].unique()



print('Embarked count', embarked_count)

print ('Embarked', unique_embarked)



train_df['Embarked'] = train_df['Embarked'].apply(convert_embarked)

sns.factorplot('Embarked', 'Survived',data=train_df, order=unique_is_alone.sort())

### 3. Explore
corr = train_df.astype(float).corr()

corr
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(corr,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Title']):

    for column in columns:

        train[column] = train[column].apply(lambda x: str(x))

        test[column] = test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)

        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)

        del train[column]

        del test[column]

    return train, test
#prep for fit

train_df.drop(['PassengerId'], axis=1, inplace=True)



test_passenger_id = test_df['PassengerId'].copy()



test_df['Age'].fillna(age_count.argmax(), inplace=True)

test_df['Sex'] = test_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

test_df['isAlone'] = (test_df['SibSp'] + test_df['Parch']) == 0

test_df['Title'] = test_df['Name'].apply(get_title)

test_df['Title'] = test_df['Title'].apply(filter_title_test).apply(classify_title)

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())

test_df['Embarked'].fillna('S', inplace=True)

test_df['Embarked'] = test_df['Embarked'].apply(convert_embarked)

test_df['Name_Len'] = test_df['Name'].apply(lambda x: len(x))

test_df.drop(['PassengerId','Ticket', 'Name', 'Cabin', 'SibSp','Parch' ], axis=1, inplace=True)



train_df, test_df = dummies(train_df, test_df)



print (train_df.head())

print (test_df.head())
#prediction



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test = test_df

X_train.shape, Y_train.shape, X_test.shape



predictions = dict()

#Random Forrest

'''random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_random_forrest = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

predictions["Random Forrest"] = (acc_random_forest, Y_pred_random_forrest)'''
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_logreg = logreg.predict(X_test)



logreg.score(X_train, Y_train)

acc_reg = round(logreg.score(X_train, Y_train) * 100, 2)

predictions["Logistic Regression"] = (acc_reg, Y_pred_logreg)
# Support Vector Machines



'''svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)



svc.score(X_train, Y_train)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

predictions["SVC"] = (acc_svc, Y_pred_svc)'''
# KNeighbors



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)



knn.score(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

predictions["KNeighbors"] = (acc_knn, Y_pred_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred_gaussian = gaussian.predict(X_test)



gaussian.score(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



predictions["Gaussian Naive Bayes"] = (acc_gaussian, Y_pred_gaussian)
predictions
key_max = max(predictions, key=predictions.get)

acc, Y_pred = predictions[key_max]



print (key_max , acc)
#submission



submission = pd.DataFrame({

        "PassengerId": test_passenger_id,

        "Survived": Y_pred

    })



print(submission)

submission.to_csv('submission.csv', index=False)