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
data.survived.value_counts(normalize=True)
#TODO 可视化预测目标的分布
data.survived.hist()
plt.show()
#TODO 利用sns画出每种舱对应的幸存与遇难人数
sns.countplot(data.pclass,hue=data.survived)
plt.show()
# TODO 打印部分名字信息
data['name'].head(10)
data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
# TODO 打印name title信息
data['name_title'].value_counts()
# TODO 名字title 与幸存的关系
data['survived'].groupby(data['name_title']).mean()
# TODO 新增名字长度的变量
data['name_len']=data['name'].apply(lambda x: len(x))
# TODO 名字长度与幸存的关系
data['survived'].groupby(pd.qcut(data['name_len'],5)).mean()
# TODO 打印性别比例
data['sex'].value_counts(normalize=True)
# TODO 性别与幸存的关系
data['survived'].groupby(data['sex']).mean()
# TODO 年龄与幸存的关系
data['survived'].groupby(pd.qcut(data['age'],5)).mean()
# TODO 登船地点的分布
data['embarked'].value_counts(normalize=True)
# TODO 登船地点与幸存的关系
data['survived'].groupby(data['embarked']).mean()
# TODO 可视化登船地点与舱位的关系
sns.countplot(data['embarked'],hue=data['pclass'])
plt.show()
data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()
def name(data):
    data['name_len'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')
    return data


def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
# TODO 
# 去掉row.names, home.dest, room, ticket, boat等属性
drop_columns = ['row.names','home.dest','room','ticket','boat']
data = data.drop(drop_columns, axis=1)
data.head()
# TODO
# 利用name(), age(), embark(), dummies()等函数对数据进行变换
data = name(data)
data = age(data)
data = embark(data)
data = dummies(data)
data.head()
from sklearn.model_selection import train_test_split
from sklearn import tree

# 准备训练集合测试集， 测试集大小为0.2， 随机种子为33
trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)

# TODO 创建深度为3，叶子节点数不超过5的决策树
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(trainX, trainY)
from sklearn import metrics
def measure_performance(X, y, model, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    #TODO complete measure_performance函数
    y_pred=model.predict(X)
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
# 利用 age, sex_male, sex_female做训练
sub_columns = ['age','sex_male','sex_female']
sub_trainX = trainX[sub_columns]
sub_testX = testX[sub_columns]
sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
sub_model.fit(sub_trainX, trainY)
measure_performance(sub_testX, testY, sub_model)
import graphviz

dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 

#TODO 生成graph文件
graph =  graphviz.Source(dot_data)
#graph.render("titanic") 
#graph.view()
graph
# TODO 观察前20个特征的重要性
pd.DataFrame(list(zip(trainX.columns, model.feature_importances_.T)), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)[:20]


