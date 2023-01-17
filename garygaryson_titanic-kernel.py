# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import GridSearchCV, cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def write_to_submission_file(predicted_labels, out_file, train_num=891,

                    target='Survived', index_label="PassengerId"):

    # turn predictions into data frame and save as csv file

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(train_num + 1,

                                                  train_num + 1 +

                                                  predicted_labels.shape[0]),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

y_train = train_df['Survived']

test_df.describe()

test_df.head()

test_df['Pclass'].plot.hist()
#test_df['Age'].plot.hist()

for data in [test_df, train_df]:

    data['Age'].fillna(data['Age'].median(), inplace=True)

    data['Embarked'].fillna(data['Embarked'].mode(), inplace=True)

    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    #data['FamilySize'] = data['Sibsp'] + data['Parch'] + 1

    

train_df = pd.concat([train_df, pd.get_dummies(train_df['Pclass'], 

                                               prefix="PClass"),

                      pd.get_dummies(train_df['Sex'], prefix="Sex"),

                      pd.get_dummies(train_df['SibSp'], prefix="SibSp"),

                      pd.get_dummies(train_df['Parch'], prefix="Parch"),

                     pd.get_dummies(train_df['Embarked'], prefix="Embarked")],

                     axis=1)

test_df = pd.concat([test_df, pd.get_dummies(test_df['Pclass'], 

                                             prefix="PClass"),

                      pd.get_dummies(test_df['Sex'], prefix="Sex"),

                      pd.get_dummies(test_df['SibSp'], prefix="SibSp"),

                      pd.get_dummies(test_df['Parch'], prefix="Parch"),

                    pd.get_dummies(test_df['Embarked'], prefix="Embarked")],

                     axis=1)
train_df.drop(['Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 

               'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 

              axis=1, inplace=True)

test_df.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], 

             axis=1, inplace=True)
train_df.shape, test_df.shape



set(test_df.columns) - set(train_df.columns)



test_df.drop(['Parch_9'], axis=1, inplace=True)

test_df.head()
dec_tree = DecisionTreeClassifier(max_depth = 2, random_state = 17) 



dec_tree.fit(train_df, y_train)

pred = dec_tree.predict(test_df)
acc_decision_tree = round(dec_tree.score(train_df, y_train) * 100, 2)

acc_decision_tree
write_to_submission_file(pred, 'submission.csv')
export_graphviz(dec_tree, out_file='age_sal_tree.dot', filled=True)

!dot -Tpng 'age_sal_tree.dot' -o 'age_sal_tree.png'
tree_params = {'max_depth': list(range(1, 5)), 

               'min_samples_leaf': list(range(1, 5))}

tree_grid = GridSearchCV(dec_tree, tree_params, cv=5, n_jobs=-1, verbose=True)

tree_grid.fit(train_df, y_train)
tree_grid.best_params_

#tree_grid.best_score_
opt_dec_tree = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 3, random_state = 17) 



opt_dec_tree.fit(train_df, y_train)

opt_pred = opt_dec_tree.predict(test_df)

write_to_submission_file(opt_pred, 'opt_submission.csv')