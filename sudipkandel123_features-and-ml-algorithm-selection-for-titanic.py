#Loading Training and testing data to a dataframe.



train = pd.read_csv("../input/titanic/train.csv",header = 0 , dtype ={'Age' : np.float64})

test = pd.read_csv("../input/titanic/test.csv" , header = 0 ,dtype = {'Age' : np.float64})

full_data = [train,test]

PassengerId = test['PassengerId']

train.head(100)

# print(PassengerId)

# print(test.info())
# 1. Pclass - Checking people who selects the particular class and mean Survived.



#print(train[['Pclass','Survived']].groupby(['Pclass']).mean())



# 2. Sex - Mean amount of male and female who survived



#print(train[['Sex','Survived']].groupby("Sex").mean())



# 3. Sibling and Spouse - SibSp && Parent and Children - Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

#print(train[['FamilySize','Survived']].groupby('FamilySize').mean())



#-------Is alone or not? -----------

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1 , 'IsAlone'] = 1

#print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    

# 4. Embarked - Fill the Missing values with most occured value 'S'2

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#print(train[['Embarked','Survived']].groupby(['Embarked']).mean()) 



# 5. Fare - Dividing Fare into four Categories and getting the mean

for dataset in full_data:

        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'])

train['CategoricalFare'] = pd.qcut(train['Fare'],4) #divide into 4 cuts and pandas decides.

#print(train[['CategoricalFare','Survived']].groupby(['CategoricalFare']).mean())



# 6. Age - 86+177 = 263 null values detected

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    #print(age_avg , age_std , age_null_count)

    #random null values which is in range 15 - 45 (avg-std , avg_std)

    age_null_random_list = np.random.randint(age_avg - age_std , age_avg + age_std , size = age_null_count)

    #print(age_null_random_list)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategorialAge'] = pd.cut(train['Age'],5)

#print (train[['CategorialAge', 'Survived']].groupby(['CategorialAge'], as_index=False).mean())

#to test whether there is any null attribute

# 7 . Name

def get_title(name):

    for dataset in full_data:

        title_search = re.search('([A-Za-z]+)\.',name)

        #print(title_search)

        if title_search:

            return title_search.group(1)

        return ""

for dataset in full_data: 

    dataset['Title'] = dataset['Name'].apply(get_title)

#print(pd.crosstab(test['Title'],train['Sex']))

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#train['Age'].isnull().sum() # to check null values for entire dataset



train.head(3)
# Mapping values to key attributes

for dataset in full_data:

    #mapping Sex

#     dataset['Sex'] = dataset['Sex'].fillna('Test')

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1 } )

    #mapping Title

    title_mapping = {"Mr" : 1 , "Miss" : 2 , "Mrs" : 3 , "Master" : 4 , "Rare" : 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'].fillna(0)

    #Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map({ 'S' : 0 , 'C' : 1 , 'Q' : 2 })

    #Mapping Fare 

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare']

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
#Feature Selection - It is trying to drop all the Attributes from test and train dataset - PassengerId,Name,Ticket,Cabin,SibSp,Parch,FamilySize,CategoricalAge,CategoricalFare



drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)

#---INSTEAD WE WILL TRY LIKE THIS AS IT IS GIVING ERROR AGAIN AND AGAIN



keep_elements = ['Survived','Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']

train = train[keep_elements]

test = train[keep_elements]

train = train.values

test  = test.values

display(train)
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

	AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



log_cols = ["Classifier", "Accuracy"]

log 	 = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = train[0::, 1::]

y = train[0::, 0]



acc_dict = {}



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]

	

	for clf in classifiers:

		name = clf.__class__.__name__

		clf.fit(X_train, y_train)

		train_predictions = clf.predict(X_test)

		acc = accuracy_score(y_test, train_predictions)

		if name in acc_dict:

			acc_dict[name] += acc

		else:

			acc_dict[name] = acc



for clf in acc_dict:

	acc_dict[clf] = acc_dict[clf] / 10.0

	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

	log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")