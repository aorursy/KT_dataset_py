# importing train and test data into train_df and test_df dataframes

import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# printing training data information 

# (number of non-null observations, datatype)

print(train.info())

print('-'*100)

print(test.info())
# taking care of missing values

def m(data):

    d = data.copy(deep = True)

    for c in data:

        if (data[c].dtype =='int64') or (data[c].dtype =='float64') : 

            if data[c].isnull().values.any():

                m = data[c].dropna().median()

                d[c].fillna(m, inplace=True)

        else:          

            if data[c].isnull().values.any():

                m = data[c].dropna().mode()[0]

                d[c].fillna(m, inplace=True)

    return d



trm = m(train)

tsm = m(test)
# printing training data information with missing values treatment

print(trm.info())

print('-'*100)

print(tsm.info())
# preparing training data

cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

x = trm[cols]

y = trm['Survived']

xx = tsm[cols]
# defining decision tree model

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from sklearn.tree import DecisionTreeClassifier

m = DecisionTreeClassifier(criterion='entropy',

                           splitter='best',

                           max_depth=None,

                           min_samples_split=2,

                           min_samples_leaf=1,

                           min_weight_fraction_leaf=0.0,

                           max_features=None,

                           random_state=None,

                           max_leaf_nodes=None,

                           min_impurity_decrease=0.0,

                           min_impurity_split=None,

                           class_weight=None,

                           presort=False)
# scoring decision tree model

from sklearn.model_selection import cross_val_score

scores = cross_val_score(m, x, y, cv = 10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# fitting decision tree model and building predictions

m.fit(x, y)

yy = m.predict(xx) 
# preparing submission file

submission = pd.DataFrame( { 'PassengerId': test['PassengerId'] , 'Survived': yy } )

submission.to_csv('tree_model_v1.csv' , index = False )