import numpy as np

import pandas as pd

from pandas import DataFrame
DATA_HOME_DIR = "../input/"

row_data = pd.read_csv(DATA_HOME_DIR + 'train.csv', index_col=0)

test_data = pd.read_csv(DATA_HOME_DIR + 'test.csv', index_col=0)
test_ind = test_data.index



train_X = row_data[['Pclass','Sex','Age','SibSp','Parch','Cabin']]

train_y = row_data[['Survived']]

test_X= test_data[['Pclass','Sex','Age','SibSp','Parch', 'Cabin']]



all_data = pd.concat([train_X, test_X])



all_data.shape, train_y.shape
## クラスごとに分割

Pclass = pd.get_dummies(all_data['Pclass'])

Pclass.columns=['1st','2nd','3rd']
## 女性、男性、子供ごとに分割

Sex = pd.get_dummies(all_data['Sex'])



def male_female_child(passenger):

    age,sex = passenger

    if np.isnan(age):

        age = 30

    if age < 16:

        return 'child'

    else:

        return sex



Person = all_data[['Age','Sex']].apply(male_female_child,axis=1)

Person = pd.get_dummies(Person)
# 独身かそうでないかで分類

Alone = all_data.Parch + all_data.SibSp



def is_alone(alone):

    if alone > 0:

        return 0

    else:

        return 1



Alone = Alone.apply(is_alone)

Alone = pd.DataFrame(Alone)

Alone.columns = ['Alone']
def get_level(deck):

    if pd.isnull(deck):

        deck = 'CXX'

    return deck[0]



Level = all_data.Cabin.apply(get_level)

Level = pd.get_dummies(Level)
merge_data = pd.merge(Alone,Pclass,right_index=True,left_index=True)

merge_data = pd.merge(merge_data,Person,right_index=True,left_index=True)

merge_data = pd.merge(merge_data,Level,right_index=True,left_index=True)



X = merge_data[:train_X.shape[0]]

y = train_y.values.ravel()



test_X = merge_data[train_X.shape[0]:]



X.shape, y.shape, test_X.shape

# tx
# create model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)
p_survived = model.predict(test_X.values)
submission = pd.DataFrame()

submission['PassengerId'] = test_ind

submission['Survived'] = p_survived
submission.to_csv('submission.csv', index=False)