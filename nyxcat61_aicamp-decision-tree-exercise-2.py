# 必要的引入

%matplotlib inline

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv(os.path.join("../input", "titanic.csv"), sep=",")
# 打印数据基本信息

data.info()
# 观察部分数据的形式

data.head(3)
# TODO 观察预测目标的分布

data['survived'].value_counts(normalize=True)
#TODO 可视化预测目标的分布

sns.countplot(data['survived'])
#TODO 利用sns画出每种舱对应的幸存与遇难人数

sns.countplot(data['pclass'], hue=data['survived'])
# TODO 打印部分名字信息

data['name'].head()
data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0].strip('.'))
# TODO 打印name title信息

data['name_title'].value_counts()
# categorize lower count titles together

title_map = {

    'Ms':'Miss', 'Mlle':'Miss',

    'Jonkheer':'noble', 'Sir':'noble', 'Madame':'noble', 'Lady':'noble',

    'Dr':'officer', 'Rev':'officer', 'Major':'officer', 'Colonel':'officer', 'Col':'officer', 'Captain':'officer', 

    'Thomas':'none', 'Leo':'none', 'Simon':'none', 'Nikolai':'none', 'Sander':'none', 'Ernst':'none', 'Seman':'none', 

    'William':'none', 'Jenny':'none', 'Berglund':'none', 'Richard':'none', 'Mansouer':'none', 'Oscar':'none', 

    'W':'none', 'the':'none', 'Khalil':'none', 'Rene':'none', 'Barton':'none', 'Jacobsohn':'none', 'Albert':'none', 

    'Nils':'none', 'Eino':'none', 'Delia':'none', 'Hilda':'none'

}



data.replace({'name_title': title_map}, inplace=True)
# TODO 名字title 与幸存的关系

data['survived'].groupby(data['name_title']).mean()
# TODO 新增名字长度的变量

data['name_len'] = data['name'].apply(lambda x: len(x))
# TODO 名字长度与幸存的关系

data['survived'].groupby(pd.qcut(data['name_len'], 5)).mean()
# TODO 打印性别比例

data['sex'].value_counts(normalize=True)
# TODO 性别与幸存的关系

data['survived'].groupby(data['sex']).mean()
# TODO 年龄与幸存的关系

data['survived'].groupby(pd.qcut(data['age'], 5)).mean()
data['age_level'] = pd.qcut(data['age'], [0, 0.12,  1])
sns.barplot('sex', 'survived', data=data, 

            hue='age_level')
plt.subplots(1, 3, figsize=(9,3))



plt.subplot(1, 3, 1)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['pclass'] == '1st'),'age']);

plt.subplot(1, 3, 2)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['pclass'] == '2nd'),'age']);

plt.subplot(1, 3, 3)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['pclass'] == '3rd'),'age']);
plt.subplots(1, 3, figsize=(9,3))



plt.subplot(1, 3, 1)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['embarked'] == 'Cherbourg'),'age']);

plt.subplot(1, 3, 2)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['embarked'] == 'Southampton'),'age']);

plt.subplot(1, 3, 3)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['embarked'] == 'Queenstown'),'age']);
plt.subplots(1, 3, figsize=(9,3))



plt.subplot(1, 3, 1)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['name_title'] == 'Mr'),'age']);

plt.subplot(1, 3, 2)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['name_title'] == 'Miss'),'age']);

plt.subplot(1, 3, 3)

sns.distplot(data.loc[np.isnan(data['age']) == False,'age']);

sns.distplot(data.loc[(np.isnan(data['age']) == False) & (data['name_title'] == 'Master'),'age']);
# TODO 登船地点的分布

data['embarked'].value_counts()
# TODO 登船地点与幸存的关系

data['survived'].groupby(data['embarked']).mean()
# TODO 可视化登船地点与舱位的关系

sns.countplot('embarked', hue='pclass', data=data)
data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()
# home.dest has lots of null values

data['dest_flag'] = data['home.dest'].apply(lambda x: 1 if pd.isnull(x) else 0)

data['survived'].groupby(data['dest_flag']).mean()
sns.barplot('dest_flag', 'survived', hue='sex', data=data)
data['room_flag'] = data['room'].apply(lambda x: 1 if pd.isnull(x) else 0)

data['survived'].groupby(data['room_flag']).mean()
sns.barplot('room_flag', 'survived', hue='sex', data=data)
def name(data):

    data['name_len'] = data['name'].apply(lambda x: len(x))

    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0].strip('.'))

    data.replace({'name_title': title_map}, inplace=True)

    del data['name']

    return data



def age(data):

    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)

    grouped_age = data.groupby(['name_title', 'pclass'])['age']

    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))

    data['age_level'] = pd.cut(data['age'], 5, labels=[1, 2, 3, 4, 5])

    return data



def embark(data):

    data['embarked'] = data['embarked'].fillna('Southampton')

    return data



def dest(data):

    data['dest_flag'] = data['home.dest'].apply(lambda x: 1 if pd.isnull(x) else 0)

    return data



def room(data):

    data['room_flag'] = data['room'].apply(lambda x: 1 if pd.isnull(x) else 0)

    return data



def dummies(data, columns=['pclass','name_title','embarked', 'sex', 'dest_flag', 'age_level']):

    for col in columns:

        data[col] = data[col].apply(lambda x: str(x))

        new_cols = [col + '_' + i for i in data[col].unique()]

        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)

        del data[col]

    return data
# TODO

# 利用name(), age(), embark(), dummies()等函数对数据进行变换

data = name(data)

data = age(data)

data = embark(data)

data = dest(data)

data = room(data)

data = dummies(data)



data.head()
# TODO 

# 去掉row.names, home.dest, room, ticket, boat等属性

drop_columns = ['row.names', 'home.dest', 'room', 'ticket', 'boat']

data = data.drop(drop_columns, axis=1)

data.head()
from sklearn.model_selection import train_test_split

from sklearn import tree



# 准备训练集合测试集， 测试集大小为0.2， 随机种子为33

trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)



# TODO 创建深度为3，叶子节点数不超过5的决策树

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5, random_state=33)

model.fit(trainX, trainY)
from sklearn import metrics

def measure_performance(X, y, model, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):

    #TODO complete measure_performance函数

    y_pred = model.predict(X)

    if show_accuracy:

        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")

    

    if show_classification_report:

        print("Classification report")

        print(metrics.classification_report(y, y_pred), "\n")

    

    if show_confussion_matrix:

        print("Confusion matrix")

        print(metrics.confusion_matrix(y, y_pred), "\n")
# TODO 调用measure_performance 观察模型在testX, testY上的表现

measure_performance(testX, testY, model)
from sklearn.model_selection import GridSearchCV, KFold

kf = KFold(n_splits=3, shuffle=True, random_state=33)



gs = GridSearchCV(tree.DecisionTreeClassifier(random_state=33), 

                 param_grid={

                     'criterion': ['gini'],

                     'max_depth': [8],

                     'min_samples_leaf': [10]

                 },

                 cv=kf)



gs.fit(data.iloc[:,1:], data.iloc[:,0])

gs.best_score_, gs.best_params_
# 利用 age, sex_male, sex_female做训练

sub_columns = ['age', 'sex_male', 'sex_female']

sub_trainX = trainX[sub_columns]

sub_testX = testX[sub_columns]

sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)

sub_model.fit(sub_trainX, trainY)
measure_performance(sub_testX, testY, sub_model)
import graphviz



dot_data = tree.export_graphviz(gs.best_estimator_, out_file=None, feature_names=trainX.columns) 



#TODO 生成graph文件

graph = graphviz.Source(dot_data) 

#graph.render("titanic") 

#graph.view()

graph
# TODO 观察前20个特征的重要性

pd.DataFrame(list(zip(testX.columns.tolist()[:20], model.feature_importances_[:20]))).sort_values(by=1, ascending=False)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

param_grid = {'random_state': [2], 

              'n_jobs':[-1],

              'min_samples_leaf': [4],

              'max_depth': [8],

              'n_estimators': [500],

              'max_features': ['sqrt'],

              'oob_score':[True]}

rf_grid = GridSearchCV(rfc,param_grid,cv=kf,refit=True,verbose=1)

rf_grid.fit(data.iloc[:,1:], data.iloc[:,0])



rf_grid.best_score_
# try logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import warnings

warnings.filterwarnings('ignore')



lrc = LogisticRegression()

clf = make_pipeline(StandardScaler(), lrc)

param_grid = {'logisticregression__penalty': ['elasticnet'], 

              'logisticregression__tol': [1e-6],

              'logisticregression__C': [0.1],

              'logisticregression__l1_ratio':[0.2],

              'logisticregression__max_iter': [200],

              'logisticregression__warm_start': [True],

              'logisticregression__solver': ['saga'],

              'logisticregression__random_state':[2]}

lrc_grid = GridSearchCV(clf,param_grid,cv=kf,refit=True,verbose=1)

lrc_grid.fit(data.iloc[:,1:], data.iloc[:,0])



lrc_grid.best_score_