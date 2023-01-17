# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainSet = pd.read_csv('/kaggle/input/titanic/train.csv')

# testSet没有 Survived 列

testSet = pd.read_csv('/kaggle/input/titanic/test.csv')

y = trainSet['Survived']

# 备份

trainSet1 = trainSet.copy()

testSet1 = testSet.copy()

combineData = list([trainSet1, testSet1])

print(1)
# drop ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']

for data in combineData:

    data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], inplace = True)

    data['Sex'] = data['Sex'].map({'male':0, 'female':1})

# 自定义离散age方法

def age(x):

    if 0<x<=12.0 :

        return 1

    elif 12.0<x<=18.0 :

        return 2

    elif 18.0<x<=40.0 :

        return 3

    elif 40.0<x<=60.0 :

        return 4

    elif 60.0<x :

        return 5

    else:

        # NaN --> 0

        return 0

# 在  trainSet1中 Embarked列缺失地方填充众数 , 众数是S

# testSet1 中Embarked列没有缺失值

trainSet1['Embarked'] = trainSet1['Embarked'].fillna('S')

# 对trainSet1、 testSet1 离散Age、Embarked

for i in combineData:

    i['Age'] = i['Age'].map(age)

    i['Embarked'] = i['Embarked'].map({'C':0, 'Q':1, 'S':2})



# 自定义离散SibSpAndParch方法

def SibSpAndParch(x):

    if x == 0 :

        return 0

    if x == 1 :

        return 1

    if x > 1 :

        return 2



for i in combineData:

    i['farmily'] = i['SibSp'] + i['Parch'] # 合并 SibSp And Parch   

    i.drop(columns = ['SibSp', 'Parch'],inplace = True)  # drop SibSp And Parch ,保留 farmily

    i['farmily'] = i['farmily'].map(SibSpAndParch) # 对 farmily 进行分组离散处理

print(1)
# 分割 训练数据 和 标签

x = trainSet1.drop(columns = ['Survived'])

y = trainSet1['Survived']

x.head()
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.2, random_state = 1)
# 交叉验证

cv_scores = []

maxDepths = [i for i in range(2,10)]

for maxDepth in maxDepths:

    model = DecisionTreeClassifier(max_depth=maxDepth)  # 数据liang

    scores = cross_val_score(model, train_x, train_y, cv = 5)

    cv_score = scores.mean()

    print('maxDepth={}，score={:.3f}'.format(maxDepth, cv_score))

    cv_scores.append(cv_score)
depth = maxDepths[np.argmax(cv_scores)]  # 获取分数最高所对应的参数

model = DecisionTreeClassifier(max_depth=depth)  # 代入模型

model.fit(train_x,train_y)

score = model.score(test_x,test_y)  # 测试集分数

print(score)
# PassengerId

id = testSet['PassengerId']

id = id.as_matrix()

result = list(zip(id,model.predict(testSet1)))

df = pd.DataFrame(result, columns = ['PassengerId', 'Survived'])

df.to_csv('decisionTreeResult.csv', index = False) # filename

# df.to_csv('./RandomForestResult.csv', index = False)

print(df.shape)

df.head()