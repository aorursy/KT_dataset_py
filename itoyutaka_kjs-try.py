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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#名前の長さと敬称を項目として保持

def names(train, test):

    for i in [train, test]:

        i['Name_Len'] = i['Name'].apply(lambda x: len(x))

        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i['Name']

    return train, test
#Ageが欠損値であったかどうかを値として保持

#敬称ごとの年齢平均値で欠損値を補完

def age_impute(train, test):

    for i in [train, test]:

        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

        data = train.groupby(['Name_Title', 'Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return train, test
#家族人数によって

#'一人','3人まで',大家族'でクラス化



def fam_size(train, test):

    for i in [train, test]:

        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',

                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))

        del i['SibSp']

        del i['Parch']

    return train, test
#チケットの最初の文字及び文字数を項目として保持

def ticket_grouped(train, test):

    for i in [train, test]:

        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])

        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))

        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],

                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            'Low_ticket', 'Other_ticket'))

        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))

        del i['Ticket']

    return train, test
#Cabinの最初の文字を値として保持

def cabin(train, test):

    for i in [train, test]:

        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

        del i['Cabin']

    return train, test
#Cabinの後ろの数字を項目として保持

def cabin_num(train, test):

    for i in [train, test]:

        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])

        i['Cabin_num1'].replace('an', np.NaN, inplace = True)

        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)

    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    del train['Cabin_num']

    del test['Cabin_num']

    del train['Cabin_num1']

    del test['Cabin_num1']

    return train, test
#embaekedの欠損値を最もおおいSで保管

def embarked_impute(train, test):

    for i in [train, test]:

        i['Embarked'] = i['Embarked'].fillna('S')

    return train, test

#文字列データをダミー化

def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):

    for column in columns:

        train[column] = train[column].apply(lambda x: str(x))

        test[column] = test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)

        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)

        del train[column]

        del test[column]

    return train, test
train, test = names(train, test)

train, test = age_impute(train, test)

train, test = cabin_num(train, test)

train, test = cabin(train, test)

train, test = embarked_impute(train, test)

train, test = fam_size(train, test)

test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train, test = ticket_grouped(train, test)

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',

                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train
from sklearn.ensemble import RandomForestClassifier as RandomForest
#ランダムフォレストの手法を使ってモデルを作成する

#加工した学習用データと答えを引数にする

#木の数は1000

model = RandomForest(n_estimators=1000,max_features='auto').fit(train.drop('Survived',axis = 1), train['Survived'])
res = pd.DataFrame(columns=['PassengerId','Survived'])
res['PassengerId']=test['PassengerId']

res['Survived']=model.predict(test)


res.to_csv("submission.csv", index=False)