import pandas as pd

from pandas import Series,DataFrame

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline


# Read required data sets.

train_data = pd.read_csv('../input/train.csv')

test_data  = pd.read_csv('../input/test.csv')

test_df  = pd.read_csv('../input/test.csv')

print(train_data.info(), train_data.head())

print(test_data.info(), test_data.head())
train_data.drop( "Name", axis=1, inplace = True)

train_data.drop( "Ticket", axis=1, inplace = True)

train_data.drop( "PassengerId", axis=1, inplace = True)



test_data.drop( "Name", axis=1, inplace = True)

test_data.drop( "Ticket", axis=1, inplace = True)

test_data.drop( "PassengerId", axis=1, inplace = True)
train_data.Age.fillna( train_data.Age.mean(),inplace=True )

train_data.drop( "Cabin", axis=1, inplace = True)

train_data.Embarked.fillna('S',inplace=True)

print(train_data.isnull().sum())



test_data.Age.fillna( test_data.Age.mean(),inplace=True )

test_data.Fare.fillna( test_data.Fare.mean(),inplace=True )

test_data.drop( "Cabin", axis=1, inplace = True)

test_data.Embarked.fillna('S',inplace=True)
enc = LabelEncoder()

train_data.Sex = enc.fit_transform(train_data.Sex)

train_data.Embarked = enc.fit_transform(train_data.Embarked)

print(train_data.head())



test_data.Sex = enc.fit_transform(test_data.Sex)

test_data.Embarked = enc.fit_transform(test_data.Embarked)
target_train    = train_data.Survived

features_train = train_data

features_train.drop('Survived',inplace=True,axis=1)

print(features_train.head())





features_test = test_data
scaler = MinMaxScaler()

features_train[:] = scaler.fit_transform( features_train[:] )

print( features_train.head())
X_train,X_test,y_train,y_test = train_test_split(features_train,target_train,random_state=42)
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import accuracy_score,fbeta_score

from time import time



def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - sample_size: the size of samples (number) to be drawn from training set

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    

    results = {}

    

    # TODO: Fit the learner to the training data using slicing with 'sample_size'

    start = time() # Get start time

    learner.fit(X_train.iloc[:sample_size],y_train[:sample_size])

    end = time() # Get end time

    

    # TODO: Calculate the training time

    results['train_time'] = end - start

        

    # TODO: Get the predictions on the test set,

    #       then get predictions on the first 300 training samples

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train.iloc[:300])

    end = time() # Get end time

    

    # TODO: Calculate the total prediction time

    results['pred_time'] = end - start



    # TODO: Compute accuracy on the first 300 training samples

    results['acc_train'] = accuracy_score(predictions_train[:300],y_train[:300])

        

    # TODO: Compute accuracy on test set

    results['acc_test'] = accuracy_score(predictions_test,y_test)

    

    # TODO: Compute F-score on the the first 300 training samples

    results['f_train'] = fbeta_score(predictions_train[:300],y_train[:300],beta=0.5)

        

    # TODO: Compute F-score on the test set

    results['f_test'] = fbeta_score(predictions_test,y_test,beta=0.5)

       

    # Success

    print ("{} trained on {} samples.",learner.__class__.__name__, sample_size)

        

    # Return the results

    return results
# TODO: Import the three supervised learning models from sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from  sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

# TODO: Initialize the three models

clf_A = KNeighborsClassifier()

clf_B = SVC()

clf_C = GradientBoostingClassifier(random_state=30)



# clf_F =

# clf_G =



# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data

size = .80 * features_train.shape[0]

samples_1 = int(.01 * size)

samples_10 = int(.1 * size)

samples_100 = int(size)





# Collect results on the learners

results = {}

for clf in [clf_A, clf_B, clf_C]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    for i, samples in enumerate([samples_1, samples_10, samples_100]):

        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)





print(results['KNeighborsClassifier'][2]['acc_train'])

print(results['SVC'][2]['acc_train'])

print(results['GradientBoostingClassifier'][2]['acc_train'])



print(results['KNeighborsClassifier'][2]['acc_test'])

print(results['SVC'][2]['acc_test'])

print(results['GradientBoostingClassifier'][2]['acc_test'])
print(features_test.isnull().sum())

predictions = clf_C.predict(features_test)
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)