import numpy as np

import pandas as pd





data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')



data_train.head()
def transform(df):    

    df.drop(['Name','Embarked', 'Ticket'], axis=1, inplace=True) #drop unecessary columns

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 #create a family size column, from sibling and parent count columns

    df.drop(['SibSp', 'Parch'], axis=1, inplace=True) #drop sibling and parent count columns

    df['Cabin'] = df['Cabin'].fillna('UKN') #fill null Cabin entries with UKN char's

    df['Age'] = df['Age'].fillna(df['Age'].median()) #fill null Age entries with the median of Age column

    df['Age'] =  df['Age'].astype(int) #convert Age to integer

    df['Fare'] = df['Fare'].fillna(df['Fare'].median()) #repeat Age process with Fare

    df['Fare'] =  df['Fare'].astype(int)

    df['Cabin'] = df.Cabin.str[:1] #take the first character of Cabin

    return df

    

transform(data_train)

transform(data_test)
from sklearn import preprocessing



def preproc(df):

    enc = preprocessing.LabelEncoder()

    df.Sex = enc.fit_transform(df.Sex)

    df.Cabin = enc.fit_transform(df.Cabin)

    return df



preproc(data_train)

preproc(data_test)



data_train.head()
data_test.head()
import sklearn.tree as tree



features = data_train[['Pclass','Sex','Age','Fare','Cabin','FamilySize']].values

target = data_train['Survived'].values



tree_ = tree.DecisionTreeClassifier()

tree_ = tree_.fit(features, target)



print(tree_.feature_importances_)

print(tree_.score(features, target))
test_features = data_test[['Pclass','Sex','Age','Fare','Cabin','FamilySize']]



prediction = tree_.predict(test_features)



PassengerId = data_test.PassengerId.values

solution =  pd.DataFrame(prediction, PassengerId, columns = ['Survived'])



solution.head(20)
from sklearn.ensemble import RandomForestClassifier





forest = RandomForestClassifier(max_depth=10, min_samples_split=4, n_estimators=100, random_state=1)

forest_ = forest.fit(features, target)



print(forest_.feature_importances_)

print(forest_.score(features, target))
pred_forest = forest_.predict(test_features)

sol_forest = pd.DataFrame(pred_forest, PassengerId, columns = ['Survived'])



sol_forest.head(20)
from sklearn import cross_validation



cross_validation.cross_val_score(forest_, X=features, y=target, cv=10)
sol_forest.to_csv('solution.csv', index_label='PassengerId')