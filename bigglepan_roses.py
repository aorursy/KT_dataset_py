# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


from sklearn.ensemble import RandomForestRegressor



### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]



    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄

    y = known_age[:, 0]



    # X即特征属性值

    X = known_age[:, 1:]



    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)



    # 用得到的模型进行未知年龄结果预测



    predictedAges = rfr.predict(unknown_age[:, 1::])



    # 用得到的预测结果填补原缺失数据



    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 



    return df, rfr
def read_data(fname):

    data = pd.read_csv(fname)

    #去除无用特征

    data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)

    #将特征数据转换为数字

    data['Sex'] = (data['Sex'] == 'male').astype('int')#两分类、

    lables = data['Embarked'].unique().tolist()

    data['Embarked'] = data['Embarked'].apply(lambda n: lables.index(n))#多分类、



    #除年龄外空值0填充

    a = data.columns

    a = a.drop('Age')

    data[a] = data[a].fillna(0)



    #用线性回归填充年龄

    set_missing_ages(data)

    return data
train = read_data('../input/train.csv')
train.isnull().sum()
y = train['Survived'].values

X = train.drop(['Survived'], axis=1).values
#多参数选择  

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



entropy_thresholds = np.linspace(0,1,50)

gint_thresholds = np.linspace(0,0.5,50)



#设置参数矩阵

param_grid = [{'criterion':['entropy'],

              'min_impurity_split':entropy_thresholds},

               {'criterion':['gini'],

               'min_impurity_split':gint_thresholds},

              {'max_depth':range(2,10)},

              {'min_samples_split':range(2,30,2)}]



clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)

clf.fit(X, y)

print("best param: {0}\nbest score: {1}".format(clf.best_params_, 

                                                clf.best_score_))

best = clf.best_estimator_
test = read_data('../input/test.csv')
test.isnull().sum()
gender_submission = pd.read_csv('../input/gender_submission.csv')
gender_submission.head()
predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions.astype(np.int32)})

result.to_csv("../titanic.csv", index=False)