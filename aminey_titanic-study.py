# data processing

import pandas as pd



## linear algebra

import numpy as np



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

 

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix
titanic = pd.read_csv('../input/titanic/train.csv')

titanic_test = pd.read_csv('../input/titanic/test.csv')

# 拿到id先进行存储，在outputs时用到

titanic_test_id = titanic_test['PassengerId']

print(titanic.head(5))

print(titanic_test.head(5))
print("shape : ",titanic.shape)



# statistical information

print(titanic.describe())



print("information -----")

print(titanic.info())
dataset = [titanic,titanic_test]
def missing_data(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False) * 100/len(df) , 2)

    # 竖直级联

    return pd.concat([total,percent],axis = 1, keys = ['Total','Percent'])

missing_data(titanic)
missing_data(titanic_test)
# 删除列 axis = 1;  inplace, 该操作是否对元数据生效

drop_column = ['Cabin']

titanic.drop(drop_column,axis = 1, inplace = True)

titanic_test.drop(drop_column,axis = 1, inplace = True)



for data in dataset:

    data['Age'].fillna(data['Age'].median(),inplace = True)

    data['Fare'].fillna(data['Fare'].median(),inplace = True)    

    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace = True) 



missing_data(titanic)
def draw(graph):

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x() + p.get_width()/2.,height +5,height , ha ="center")
# sns.countplot 使用图形柱状图显示每个类别的 数量统计 ； 以每个value值作为x轴上的分类类型  value count for a single or two variable

# x x标签 ； hue 在x,y标签划分的同时，再以hue标签统计

# plt.figure(figsize) 表示 figure 的长和宽

sns.set(style = "darkgrid")

plt.figure(figsize = (8,5))

graph = sns.countplot(x = 'Survived',hue = 'Survived',data = titanic)

# draw(graph)
plt.figure(figsize=(8,5))

graph = sns.countplot(x= "Sex",hue = "Survived",data = titanic)
# plt.subplots()把图形分成 nrows * ncols 个子图， 把子图赋值给 ax，ax[0]为第一个子图，ax[1]第二个...

# sns.countplot(data ,ax) ,将数据绘制到子图上

fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10,4))

x = sns.countplot(titanic['Pclass'], ax = ax[0])

y = sns.countplot(titanic['Embarked'], ax = ax[1])

draw(x)

draw(y)

fig.show()
# FacetGrid() 用于 子集中分别可视化变量的分布， 或者多个变量之间的关系时，非常有用

# .FacetGrid() 画出轮廓 ； .map()填充数据

# .FacetGrid()  col,row,hue 定义子集的变量，在网格的不同方面绘制；放在顶部，展示子集数据  ； aspect 每个小图的横轴和纵轴的比例

# FacetGrid .add_legend() 绘制一个图例

# FacetGrid .map(func , *args, *kwargs) 将绘图应用于每个方面的数据子集

# sns.pointplot 点图； 

FacetGrid = sns.FacetGrid(titanic , col = 'Pclass' , height = 4,aspect = 1)

FacetGrid.map(sns.pointplot, 'Embarked','Survived','Sex',order = None, hue_order = None)

FacetGrid.add_legend()
# .drop() 删除行或列 ； 默认删除某行，axis = 1，删除某列； inplace = True 在原数据上改变

drop_column = ['Embarked']

titanic.drop(drop_column,axis = 1 ,inplace = True)

titanic_test.drop(drop_column,axis = 1 ,inplace = True)
plt.figure(figsize = (8,5))

pclass = sns.countplot(x = 'Pclass' , hue = 'Survived',data = titanic)
plt.figure(figsize = (8,5))

sns.barplot(x = 'Pclass', y = 'Survived',data = titanic)
all_data = [titanic,titanic_test]

for dataset in all_data:

    dataset['Family'] = dataset['SibSp'] + dataset['Parch']+1
# sns.factorplot 绘制多个维度的变量 factorplot(x,y,data,aspect)   

axes = sns.factorplot('Family','Survived',data = titanic ,aspect = 2.5)
axes = sns.factorplot('Family','Age','Survived',data = titanic , aspect = 2.5,)
# pd.cut() 分箱操作 bins:标量序列或间隔索引，是分箱的依据 ； labels：分箱的标签

for dataset in all_data:

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins = [0,12,20,40,120],labels = ['children','teens','adult','elder'])

plt.figure(figsize = (8,5))

sns.barplot(x = 'Age_bin' , y = 'Survived',data = titanic)
plt.figure(figsize = (8,5))

age = sns.countplot(x = 'Age_bin',hue='Survived',data = titanic)
AAS = titanic[['Sex','Age_bin','Survived']].groupby(['Sex','Age_bin'],as_index = False).mean()

sns.factorplot('Age_bin', 'Survived','Sex',data = AAS,aspect = 3,kind='bar')

plt.suptitle('Age,Sex vs Survived')
for dataset in all_data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'],bins = [0,10,50,100,550],labels = ['low_fare','mid_fare','ave_fare','high_fare'])

plt.figure(figsize = (8,5))

sns.countplot(x = 'Pclass',hue = 'Fare_bin', data = titanic)
sns.barplot(x = 'Fare_bin', y ='Survived', data = titanic)
# abs() 绝对值    

# corr() 相关系数矩阵，给出了任意两个变量之间的相关系数 corr()[A]只显示A和其他变量之间的相关系数

# 上三角和下三角堆成

pd.DataFrame(abs(titanic.corr()['Survived']).sort_values(ascending = False))
# np.zero_like（W，dtype = np.bool） 生成一个 初始化为 False 的bool型矩阵，维度和 w 一致

# np.triu_indices_from 返回数组上三角的索引

# mask[np.triu_indices_from(mask)] = True 将上三角区域置为True

# 以上 生成了一个掩码

# sns.heatmap() 生成热力图 

# sns.heatmap(data , vmax/vmin = 设置颜色带的最大最小值，cmap = 设置颜色带的色系， 

#  center = 设置颜色带的分界线，annot = 是否显示数值注释，fmt = 数值的格式化形式，

#  linewidths/linecolor = 每个小方格之间的间距和颜色，map = 传入布尔型矩阵，为true的地方，热力图相应的区域会被屏蔽)

corr = titanic.corr()



mask = np.zeros_like(corr, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(12,8))

sns.heatmap(corr,

            annot=True,

            mask = mask,

            cmap = 'RdBu',

            linewidths = .9,

            linecolor = 'white',

            vmax = 0.3,

            fmt = '.2f',

            center = 0,

            square = True)

plt.title('Correlations Metrix', y = 1, fontsize = 20, pad = 20)
titanic.info()
gender = {"male":0 , "female":1}

for data in all_data:

    data['Sex'] = data['Sex'].map(gender)

titanic['Sex'].value_counts()
for dataset in all_data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[dataset['Age']<=15,'Age'] = 0

    dataset.loc[(dataset['Age']>15) & (dataset['Age']<=20),'Age'] = 1

    dataset.loc[(dataset['Age']>20) & (dataset['Age']<=26),'Age'] = 2    

    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=28),'Age'] = 3     

    dataset.loc[(dataset['Age']>28) & (dataset['Age']<=35),'Age'] = 4     

    dataset.loc[(dataset['Age']>35) & (dataset['Age']<=45),'Age'] = 5  

    dataset.loc[dataset['Age']>45,'Age'] = 6

titanic['Age'].value_counts()
# for dataset in [titanic]:

#     drop_column = ['Age_bin','Fare','Name','Ticket','PassengerId','SibSp','Parch','Fare_bin']

# #     drop_column = ['Age_bin','Fare','Name','Ticket','SibSp','Parch','Fare_bin']

#     dataset.drop(drop_column ,axis = 1,inplace = True)

    

# for dataset in [titanic_test]:

# #     drop_column = ['Age_bin','Fare','Name','Ticket','PassengerId','SibSp','Parch','Fare_bin']

#     drop_column = ['Age_bin','Fare','Name','Ticket','SibSp','Parch','Fare_bin']

#     dataset.drop(drop_column ,axis = 1,inplace = True)

    

for dataset in all_data:

    drop_column = ['Age_bin','Fare','Name','Ticket','PassengerId','SibSp','Parch','Fare_bin']

#     drop_column = ['Age_bin','Fare','Name','Ticket','SibSp','Parch','Fare_bin']

    dataset.drop(drop_column ,axis = 1,inplace = True)

    

    

# titanic   DataFrame

# [titanic].dtype()  list
titanic.info()
# train_test_split( train_data , train_target , test_size , random ) 随机划分样本为训练集和测试集

# train_data 待划分的样本数据，  train_target 待划分样本数据的结果【标签】

# test_size 测试数据占样本数据的比例，若为整数则代表样本数量。     random 随机数种子，保证每次都是同一个随机数

# X_train,y_train 得到的训练数据  X_test,y_test 得到的测试数据  

all_features = titanic.drop('Survived',axis = 1)

Target = titanic['Survived']

X_train,X_test,y_train,y_test = train_test_split(all_features,Target,test_size = 0.3 , random_state = 0)

X_train.shape , X_test.shape , y_train.shape , y_test.shape
# 逻辑回归。 回归模型分为线性回归，处理 因变量是连续变量的问题。 逻辑回归 处理 因变量是分类变量 [两点（0-1）分布变量]

# sklearn逻辑回归算法：LogisticRegression() 导入模型 ； fit(x,y) 训练模型 ； predict() 预测，用训练得到的模型对数据集进行预测，返回预测结果



# accuracy_score(y_true , y_pred , normalize 默认True返回正确分类的比例-为False返回正确分类的样本数)

# 评估方法，分类准确率分时 ； 所有分类正确的百分比 

# round() 格式化数据，按照指定的小数位数四舍五入

# KFold() K折交叉验证   n_splits 最少划分为几块 ； random_state 随机种子数

#   将训练集划分为n_splits块互斥子集，每次用其中一个子集作为验证集，其他子集作为训练集，进行n_splits次训练，得到n_splits个结果

# cross_val_score() sklearn 提供的交叉验证方法，KFold 划分数据集，cross_val_score 根据模型进行计算，即调用了KFold进行数据集划分

model = LogisticRegression()

model.fit(X_train , y_train)

prediction_lr = model.predict(X_test)

Log_acc = round(accuracy_score(prediction_lr,y_test)*100,2)



# kfold = KFold(n_splits = 10 , random_state = 22)   # 有点迷惑为啥调用KFold, cross_val_score默认调用了KFold划分数据集了，挠头

log_cv_acc = cross_val_score(model , all_features , Target , cv = 10 , scoring = 'accuracy')



print('The accuracy of the Logistic Regression is ',Log_acc)

print('The cross validated score for Logistic Regression is ',round(log_cv_acc.mean()*100,2))
# KNN K临近算法 ；  临近算法：将测试图片和 训练集图片一一计算相似度，相似度最高图片的标签，即是测试图片的标签

# K 临近算法 可以找 k 个最相近的图片，再用k张图中数量最多的标签，作为测试图片的标签

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train,y_train)*100 , 2)



# kflod = KFold(n_splits = 10, random_state = 22)

result_knn = cross_val_score(model , all_features , Target , cv = 10 , scoring = 'accuracy')



print('The accuracy of the K Nearst Neighbors Classifier is ',acc_knn)

print('The cross validated score for K Nearst Neighbors Classifier is ',round(result_knn.mean()*100,2))
'''

高斯朴素贝叶斯

朴素贝叶斯： 基于概率理论，假定 属性相互条件独立。 p(类别/特征) = p(类别,特征)/p(类别) ； 在属性个数多，或者属性之间相关性较大时，分类效果不好

高斯朴素贝叶斯： 假设每个分类相关的连续值是按照高斯分布的

'''

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)

pred_gnb = model.predict(X_test)

gnb_acc = round(accuracy_score(pred_gnb , y_test)*100,2)



result_gnb = cross_val_score(model , all_features , cv = 12 , scoring = 'accuracy')



print('The accuracy of the Gaussian Naive Bayes is ',gnb_acc)

print('The cross validated score for Gaussian Naive Bayes is ',round(result_gnb.mean()*100,2))
'''

线性支持向量机

向量机 SVM，//TODO

'''

linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)

pred_svc = linear_svc.predict(X_test)

svc_acc = round(linear_svc.score(X_train,y_train)*100,2)

result_svc = cross_val_score(model,all_features,Target,cv = 10 , scoring = 'accuracy')



print('The accuracy of the Linear Support Vector Machine is ',svc_acc)

print('The cross validated score for Linear Support Vector Machine is ',round(result_svc.mean()*100,2))
random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train , y_train)

pred_rf1 = random_forest.predict(X_test)

acc_rf = round(random_forest.score(X_train,y_train)*100 , 2)



result_rf = cross_val_score(model,all_features , Target,cv = 10,scoring = 'accuracy')



print('The accuracy of the Random Forest is ',acc_rf)

print('The cross validated score for Random Forest is ',round(result_rf.mean()*100,2))
'''

决策树，是树形模型，自顶向下递归，以信息熵为度量构造一棵熵值下降最快的树

和随机森林的区别：随机森林是森林，决策树是树； 决策树进行剪枝 ； 

'''

from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train,y_train)

pred_dt = decision_tree.predict(X_test)

acc_dt = round(decision_tree.score(X_train,y_train)*100 , 2)



result_dt = cross_val_score(model,all_features,Target , cv = 10,scoring = 'accuracy')



print('The accuracy of the Decision Tree is ',acc_dt)

print('The cross validated score for Decision Tree is ',round(result_dt.mean()*100,2))
result = pd.DataFrame({

    'Model' : ['Support Vector Machine','KNN','LogisticRegression',

              'Random Forest','Gaussian Naive Bayes', 'Decision Tree'],

    'Score' : [svc_acc, acc_knn,Log_acc,acc_rf,gnb_acc,acc_dt]

})

result_df = result.sort_values(by = 'Score',ascending = False)

result_df = result_df.set_index('Model')

result_df.head(9)
'''

输出结果并不固定

cross_val_score 分类准确率，对数据集进行k次交叉验证，并为每次验证结果评测，返回模型的评测结果

cross_val_predict 返回分类结果

confusion_matrix 混淆矩阵，总结分类模型的预测结果的分析表

精确率Precision=TP/(TP+FP),召回率recall=TP/(TP+FN),准确率accuracy=(TP+FN+FP+TN)

TN  FP

FN  TP

'''

pred_rf = cross_val_predict(random_forest , X_train , y_train , cv = 3)

confusion_matrix(y_train,pred_rf)
'''

精确度  P = TP/TP+FP   分类器判定的正例中，正样本占的比例  

召回率  R = TP/TP+FN   预测为正的比例，占正例总数的比例

'''

from sklearn.metrics import precision_score,recall_score

print("Precision : ",precision_score(y_train,pred_rf))

print("Recall : ",recall_score(y_train,pred_rf))
'''

f-score 是精确率和召回率的调和平均数

f1 = ((精确率*召回率)/精确率+召回率）*2  

'''

from sklearn.metrics import f1_score

f1_score(y_train, pred_rf)
from sklearn.metrics import roc_curve

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]

FPR, TPR , thresholds = roc_curve(y_train,y_scores)

def plot_roc_curve(FPR,TPR,label = None):

    plt.plot(FPR,TPR,linewidth = 2 , label = label)

    plt.plot([0,1],[0,1],'r',linewidth = 4)

    plt.axis([-0.05,1.05,-0.05,1.05])

    plt.xlabel('FPR',fontsize = 16)

    plt.ylabel('TPR',fontsize = 16)    



plt.figure(figsize=(14,7))

plot_roc_curve(FPR,TPR)

plt.show()
titanic_test.info()
# 模型已训练好-直接预测即可

features = ['Pclass','Sex','Age','Family']



test_show_data=pd.get_dummies(titanic_test[features])

# test_show_data=pd.get_dummies(titanic_test,columns = ['Pclass','Sex','Age','Family'])

pre_test = random_forest.predict(test_show_data)

output = pd.DataFrame({'PassengerId': titanic_test_id, 'Survived': pre_test})

output.to_csv('titanic_predict2.csv', index=False)

print("Your submission was successfully saved!")
titanic_test.columns
pd.get_dummies(titanic_test,columns = ['Pclass','Sex','Age','Family'])