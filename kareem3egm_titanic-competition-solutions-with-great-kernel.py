# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# import library for machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
train_df.head()
print(train_df.columns.values)
print("train shape:",train_df.shape)

print("test shape :",test_df.shape)

train_df.info()

print('_______________________________________________')

test_df.info()
train_df.isnull().sum()



from pandas_profiling import ProfileReport 



profile = ProfileReport( train_df, title='Pandas profiling report ' , html={'style':{'full_width':True}})



profile.to_notebook_iframe()

train_df.describe()

train_df.describe(include=['O'])
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=4.2, aspect=2)

grid.map(plt.hist, 'Age', alpha=.75, bins=40)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


#Applying VotingClassifier Model 



'''

#ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)

'''



#loading models for Voting Classifier



LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=33)



GBCModel_ = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=7, random_state=33)



DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=10,random_state = 33)



RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=10, random_state=33)



SVCModel_ = SVC(kernel = 'rbf', random_state = 33,C = 0.1,degree = 2)



GaussianNBModel_ = GaussianNB()



KNNModel_ = KNeighborsClassifier(n_neighbors= 4 , weights ='uniform', algorithm='auto')



NNModel_ = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(100, 3),learning_rate='constant',activation='tanh')



SGDModel_ = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)



#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[('LRModel',LRModel_),('GBCModel',GBCModel_),('DTModel',DTModel_),('RFModel',RFModel_),('SVCModel',SVCModel_),('GaussianNBModel',GaussianNBModel_),('KNNModel',KNNModel_),('NNModel',NNModel_),('SGDModel',SGDModel_)], voting='hard')



VotingClassifierModel.fit(X_train, Y_train)



#Calculating Details

print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, Y_train))



print('----------------------------------------------------')





#Calculating Prediction

Y_predtrain = VotingClassifierModel.predict(X_train)

print('Predicted Value for VotingClassifierModel is : ' , Y_predtrain[:10])





AccScore = accuracy_score(Y_train,Y_predtrain, normalize=False)

print('Accuracy Score is : ', AccScore)




F1Score = f1_score(Y_train,Y_predtrain, average='micro') #it can be : binary,macro,weighted,samples

print('F1 Score is : ', F1Score)


#Calculating Confusion Matrix

CM = confusion_matrix(Y_train,Y_predtrain)

print('Confusion Matrix is : \n', CM)



# drawing confusion matrix

sns.heatmap(CM, center = True)

plt.show()




#Calculating Prediction

Y_pred = VotingClassifierModel.predict(X_test)

print('Predicted Value for VotingClassifierModel is : ' , Y_pred[:10])





submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred })



submission.to_csv('mysubmission.csv',index=False)