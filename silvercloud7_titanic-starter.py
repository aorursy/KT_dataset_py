import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# lets grab the training data, test data and example submission that assumes all women survive using pandas

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

example_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.info()
train.head()
test.head()
test.info()
print('Train columns with null values:\n', train.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', test.isnull().sum())

print("-"*10)



train.describe(include = 'all')
datasets = [ train, test]





for dataset in datasets:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

    dataset['Family'] =  dataset['SibSp'] + dataset['Parch'] + 1 

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

        

    dataset['IsAlone'].loc[dataset['Family'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

#     dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print('Train columns with null values:\n', train.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', test.isnull().sum())

print("-"*10)



train.describe(include = 'all')
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



male = train.loc[train.Sex == 'male']["Survived"]

rate_male = sum(male)/len(male)



print("% of male who survived:", rate_male)

train.hist(figsize=(8,8))
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]



features = ["Pclass", "Sex","Age","Family","IsAlone"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# quick way to test your model accuracy without having to submit check average 



from sklearn.model_selection import cross_val_score



scores = cross_val_score(model,X,y,cv=10)



def displayCVScores(scores):

    print("Scores: ",scores)

    print("Mean:",scores.mean())

    print("Standard Deviation:",scores.std())

displayCVScores(scores)
# feature importance

importances = pd.DataFrame({'feature':X.columns,'importance':np.round(model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(10)
plot = importances.plot.pie(y='importance', figsize=(10, 10))


import sklearn.tree 

import graphviz 



# Extract single tree

estimator = model.estimators_[10]



dot_data = sklearn.tree.export_graphviz(estimator, out_file=None, 

               feature_names=X.columns,  

                class_names=['Survive','Dies'] , filled=True, rounded=True,  special_characters=True)  

graph = graphviz.Source(dot_data) 



graph