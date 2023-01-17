# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn import model_selection

from sklearn.linear_model import LogisticRegressionCV

from tqdm import tqdm
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
train_data.info()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data.Sex.value_counts()
train_data.Pclass.value_counts()
train_data['Embarked'].value_counts()

common_value = 'S'

train_data.Embarked.fillna(common_value, inplace=True)
train_data["Alone"] = [1 if (x + y) == 0 else 0 for x, y in zip(train_data.SibSp, train_data.Parch)]

test_data["Alone"] = [1 if (x + y) == 0 else 0 for x, y in zip(test_data.SibSp, test_data.Parch)]



#Creating new family_size column

train_data['Family_Size']=train_data['SibSp'] + train_data['Parch']

test_data['Family_Size']=test_data['SibSp'] + test_data['Parch']
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if str.find(big_string, substring) != -1:

            return substring

    print(big_string)

    return np.nan



title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']



#Now that I have them, I recombine them to the four categories.



train_data['Title'] = train_data['Name'].map(lambda x: substrings_in_string(x, title_list))

test_data['Title'] = test_data['Name'].map(lambda x: substrings_in_string(x, title_list))



#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

train_data['Title'] = train_data.apply(replace_titles, axis=1)

test_data['Title'] = test_data.apply(replace_titles, axis=1)
#Turning cabin number into Deck

train_data.Cabin.fillna("Unknown", inplace=True)

test_data.Cabin.fillna("Unknown", inplace=True)

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_data['Deck'] = train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

test_data['Deck'] = test_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
train_data.Age.fillna(train_data.Age.mean(), inplace=True)

test_data.Age.fillna(test_data.Age.mean(), inplace=True)

train_data['Age*Class'] = train_data['Age'] * train_data['Pclass']

test_data['Age*Class'] = test_data['Age'] * test_data['Pclass']
#No fare available

test_data.loc[:, "Fare"].fillna(test_data[test_data.Pclass == 3]["Fare"].mean(), inplace=True)



train_data['Fare_Per_Person'] = train_data['Fare']/(train_data['Family_Size'] + 1)

test_data['Fare_Per_Person'] = test_data['Fare']/(test_data['Family_Size'] + 1)
train_data.head(10)
y = train_data["Survived"]



features = ["Pclass", "Sex", "Family_Size", "Title", "Age*Class", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

#No Deck T in test set

#X_test['Deck_T'] = 0



#cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 )

#grid_criterion = ['gini', 'entropy']

#grid_max_depth = [2, 4, 6, 8, 10, None]

#grid_n_estimator = [10, 50, 100, 300]

#grid_seed = [0]



#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1)

#model = model_selection.GridSearchCV(RandomForestClassifier(),

#                                          param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion,

#                                                      'max_depth': grid_max_depth, 'random_state': grid_seed}, 

#                                          scoring = 'roc_auc', cv = cv_split)

model.fit(X, y)

predictions = model.predict(X_test)

#print('Best Parameters: ', model.best_params_)

print("Accuracy, Training Set: {:.2%}".format(accuracy_score(y, model.predict(X))))



#Get Performance by Class (Lookup Confusion Matrix)

pd.crosstab(y, model.predict(X), margins=True, rownames=['Actual'], colnames=['Predicted'])
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#Fit an XGBoost Model



#Training

model = XGBClassifier()

model.fit(X, y)

predictions = model.predict(X_test)



#Check Accuracy of Spam Detection in Train and Test Set

print("Accuracy, Training Set: {:.2%}".format(accuracy_score(y, model.predict(X))))

#Get Performance by Class (Lookup Confusion Matrix)

pd.crosstab(y, model.predict(X), margins=True, rownames=['Actual'], colnames=['Predicted'])
#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")
#Find optimal depth of trees

depth, tree_start, tree_end = {}, 3, 20

for i in range(tree_start, tree_end):

    model = DecisionTreeClassifier(max_depth=i)

    scores = cross_val_score(estimator=model, X=X, y=y, cv=5, n_jobs=-1)

    depth[i] = scores.mean()

    

#Plot

lists = sorted(depth.items())

x_axis, y_axis = zip(*lists) 

plt.ylabel("Cross Validation Accuracy")

plt.xlabel("Maximum Depth")

plt.title('Variation of Accuracy with Depth - Simple Decision Tree')

plt.plot(x_axis, y_axis, 'b-', marker='o')

plt.show()
#Make best depth a variable

best_depth = sorted(depth, key=depth.get, reverse=True)[0]

print("The best depth was found to be:", best_depth)
#Evaluate the performance at the best depth

model = DecisionTreeClassifier(max_depth=best_depth)

model.fit(X, y)

predictions = model.predict(X_test)



#Check Accuracy of Spam Detection in Train and Test Set

print("Accuracy, Training Set: {:.2%}".format(accuracy_score(y, model.predict(X))))

#Get Performance by Class (Lookup Confusion Matrix)

pd.crosstab(y, model.predict(X), margins=True, rownames=['Actual'], colnames=['Predicted'])
#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")
#Find Optimal Depth of trees for Boosting

score_train, score_test, depth_start, depth_end = {}, {}, 2, 20

for i in tqdm(range(depth_start, depth_end)):

    model = AdaBoostClassifier(

        base_estimator=DecisionTreeClassifier(max_depth=i),

        n_estimators=100, learning_rate=0.05)

    model.fit(X, y)

    score_train[i] = accuracy_score(y, model.predict(X))
#Plot

lists1 = sorted(score_train.items())

x1, y1 = zip(*lists1) 

plt.ylabel("Accuracy")

plt.xlabel("Depth")

plt.title("Optimal Depth of trees for Boosting")

plt.plot(x1, y1, 'b-', label='Train')

plt.legend()

plt.show()
#Fit an Adaboost Model



#Training

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, learning_rate=0.05)

model.fit(X, y)

predictions = model.predict(X_test)



#Check Accuracy of Spam Detection in Train and Test Set

print("Accuracy, Training Set: {:.2%}".format(accuracy_score(y, model.predict(X))))

#Get Performance by Class (Lookup Confusion Matrix)

pd.crosstab(y, model.predict(X), margins=True, rownames=['Actual'], colnames=['Predicted'])
#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")