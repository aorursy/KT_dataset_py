# 必要的引入
%matplotlib inline
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(os.path.join("../input/titanic", "train.csv"), sep=',')
# 打印数据基本信息
data.info()
# 观察部分数据的形式
data.head(3)
# TODO 观察预测目标的分布
print(data.Survived.mean())
#TODO 可视化预测目标的分布
sns.countplot(data.Survived)
#TODO 利用sns画出每种舱对应的幸存与遇难人数
sns.countplot(data.Pclass,hue=data.Survived)
# TODO 打印部分名字信息
print(data.Name[:10])
data['name_title'] = data['Name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
# TODO 打印name title信息
print(data.name_title.value_counts())
# TODO 名字title 与幸存的关系
data.Survived.groupby(data.name_title).describe()

# TODO 新增名字长度的变量
data['name_len']=data.Name.apply(lambda x:len(x))
data.name_len.head()
# TODO 名字长度与幸存的关系
data['name_len_level'] = pd.qcut( data['name_len'], 10 )
data.Survived.groupby(data.name_len_level).mean()
# TODO 打印性别比例
print(data.Sex.value_counts(normalize=True))
# TODO 性别与幸存的关系
sns.countplot(data.Sex,hue=data.Survived)
# TODO 年龄与幸存的关系
data['age_level'] = pd.qcut( data['Age'], 5 )
data.Survived.groupby(data.age_level).mean()
# TODO 登船地点的分布
print(data.Embarked.value_counts(normalize=True))
# TODO 登船地点与幸存的关系
data.Survived.groupby(data.Embarked).mean()
# TODO 可视化登船地点与舱位的关系
sns.countplot(data.Embarked,hue=data.Pclass)
print(data.Survived.groupby(data.SibSp).mean())
sns.countplot(data.SibSp,hue=data.Survived)

def name(data):
    data['name_len'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'Pclass'])['Age']
    data['Age'] = grouped_age.transform(lambda x: x.fillna(data['Age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

def embark(data):
    data['Embarked'] = data['Embarked'].fillna('Southampton')
    return data


def dummies(data, columns=['Pclass','name_title','Embarked', 'Sex','SibSp']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
data.head()
# TODO 
# 去掉row.names, home.dest, room, ticket, boat等属性
drop_columns = ['PassengerId','Name','Parch','Ticket','Fare','Cabin','name_len_level','age_level']
data = data.drop(columns=drop_columns)
data.head()
# TODO
# 利用name(), age(), embark(), dummies()等函数对数据进行变换
data = age(data)
data = embark(data)
data = dummies(data, columns=['Pclass','name_title','Embarked', 'Sex','SibSp'])
data.head()
from sklearn.model_selection import train_test_split
from sklearn import tree

# 准备训练集合测试集， 测试集大小为0.2， 随机种子为33
trainX, testX, trainY, testY = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size=0.2, random_state=33)
# TODO 创建深度为3，叶子节点数不超过5的决策树
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(trainX, trainY)
y_pred=model.predict(testX)

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
measure_performance(testX, testY,model)
# 利用 age, sex_male, sex_female做训练
sub_columns = ['Age','Sex_male','Sex_female']
sub_trainX = trainX[sub_columns]
sub_testX = testX[sub_columns]
sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
sub_model.fit(sub_trainX, trainY)
measure_performance(sub_testX, testY, sub_model)
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 

#TODO 生成graph文件
graph = graphviz.Source(dot_data) 
 
#graph.render("titanic") 
#graph.view()
graph
# TODO 观察前20个特征的重要性
print(len(trainX.columns))
pd.DataFrame(model.feature_importances_, columns=['importance'])
#pd.DataFrame(trainX.columns,columns=['variables'])
pd.concat([pd.DataFrame(trainX.columns,columns=['variables']),pd.DataFrame(model.feature_importances_, columns=['importance'])],axis=1).sort_values(by='importance',ascending=False)[:20]



