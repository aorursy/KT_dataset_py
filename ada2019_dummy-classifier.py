# importing train and test data into train_df and test_df dataframes

import pandas as pd

train = pd.read_csv('/kaggle/input/sce-data-science-2020-course/train.csv')

test = pd.read_csv('/kaggle/input/sce-data-science-2020-course/test.csv')
# printing training data information 

# (number of non-null observations, datatype)

print(train.info())

print('-'*100)

print(test.info())
# preparing training data

trm = train.copy(deep = True)

tsm = test.copy(deep = True)

cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

x = trm[cols]

y = trm['Survived']

xx = tsm[cols]
# defining dummy model

#https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

from sklearn.dummy import DummyClassifier

m = DummyClassifier(strategy='most_frequent')
# scoring decision tree model

from sklearn.model_selection import cross_val_score

scores = cross_val_score(m, x, y, cv = 10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# fitting decision tree model and building predictions

m.fit(x, y)

yy = m.predict(xx) 
# preparing submission file

submission = pd.DataFrame( { 'PassengerId': test['PassengerId'] , 'Survived': yy } )

submission.to_csv('dummy_model_v1.csv' , index = False )