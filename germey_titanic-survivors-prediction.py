# 导入库
# 数据分析和探索
import pandas as pd
import numpy as np
import random as rnd

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# 获取数据，训练集train_df，测试集test_df，合并集合combine（便于对特征进行处理时统一处理：for df in combine:）
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df
# 探索数据
# 查看字段结构、类型及head示例
train_df.head(10)
# 查看各特征非空样本量及字段类型
train_df.info()
print("_"*40)
test_df.info()
# 查看数值类（int，float）特征的数据分布情况
train_df.describe()
# 查看非数值类（object类型）特征的数据分布情况
train_df.describe(include=["O"])
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 富人和中等阶层有更高的生还率，底层生还率低
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 性别和是否生还强相关，女性用户的生还率明显高于男性
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 有0到2个兄弟姐妹或配偶的生还几率会高于有更多的
train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)
# 同行的父母或孩子总数相关
g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Age",bins=20)
# 婴幼儿的生存几率更大
# Fare
g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Fare",bins=10)
# 票价最便宜的幸存几率低
grid = sns.FacetGrid(train_df,row="Survived",col="Sex",aspect=1.6)
grid.map(plt.hist,"Age",alpha=.5,bins=20)
grid.add_legend()
# 女性的幸存率更高，各年龄段均高于50%
# 男性中只有婴幼儿幸存率高于50%，年龄最大的男性（近80岁）幸存
grid1 = sns.FacetGrid(train_df,col="Embarked")
grid1.map(sns.pointplot,"Pclass","Survived","Sex",palette = "deep")
#
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# Some features of my own that I have added in
# Gives the length of the name
train_df['NameLength'] = train_df['Name'].apply(len)
test_df['NameLength'] = test_df['Name'].apply(len)
train_df
# Feature that tells whether a passenger had a cabin on the Titanic
train_df['HasCabin'] = train_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_df['HasCabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
train_df
# 剔除Ticket（人为判断无关联）和Cabin（有效数据太少）两个特征
train_df = train_df.drop(["Ticket","Cabin"],axis=1)
test_df = test_df.drop(["Ticket","Cabin"],axis=1)
combine = [train_df,test_df]
print(train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
# 根据姓名创建称号特征，会包含性别和阶层信息
# dataset.Name.str.extract(' ([A-Za-z]+)\.' -> 把空格开头.结尾的字符串抽取出来
# 和性别匹配，看各类称号分别属于男or女，方便后续归类

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by=["male","female"],ascending=False)
# 把称号归类为Mr,Miss,Mrs,Master,Rare_Male,Rare_Female(按男性和女性区分了Rare)
for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Dona'],"Rare_Female")
    dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Don','Dr','Major',
                                                 'Rev','Sir','Jonkheer',],"Rare_Male")
    dataset["Title"] = dataset["Title"].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')
dataset
# 按Title汇总计算Survived均值，查看相关性
train_df[["Title","Survived"]].groupby(["Title"],as_index=False).mean()
# title特征映射为数值
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare_Female":5,"Rare_Male":6}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)
    # 为了避免有空数据的常规操作
train_df.head()
# Name字段可以剔除了
# 训练集的PassengerId字段仅为自增字段，与预测无关，可剔除
train_df = train_df.drop(["Name","PassengerId"],axis=1)
test_df = test_df.drop(["Name"],axis=1)
# 每次删除特征时都要重新combine
combine = [train_df,test_df]
combine[0].shape,combine[1].shape
# sex特征映射为数值
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"female":1,"male":0}).astype(int)
    # 后面加astype(int)是为了避免处理为布尔型？
train_df.head()
# 对Age字段的空值进行预测补充
# 取相同Pclass和Title的年龄中位数进行补充（Demo为Pclass和Sex）

grid = sns.FacetGrid(train_df,col="Pclass",row="Title")
grid.map(plt.hist,"Age",bins=20)
guess_ages = np.zeros((6,3))
guess_ages
# 给age年龄字段的空值填充估值
# 使用相同Pclass和Title的Age中位数来替代（对于中位数为空的组合，使用Title整体的中位数来替代）


for dataset in combine:
    # 取6种组合的中位数
    for i in range(0, 6):
        
        for j in range(0, 3):
            guess_title_df = dataset[dataset["Title"]==i+1]["Age"].dropna()
            
            guess_df = dataset[(dataset['Title'] == i+1) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median() if ~np.isnan(guess_df.median()) else guess_title_df.median()
            #print(i,j,guess_df.median(),guess_title_df.median(),age_guess)
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    # 给满足6中情况的Age字段赋值
    for i in range(0, 6):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == i+1) & (dataset.Pclass == j+1),
                        'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
#创建是否儿童特征
for dataset in combine:
    dataset.loc[dataset["Age"] > 12,"IsChildren"] = 0
    dataset.loc[dataset["Age"] <= 12,"IsChildren"] = 1
train_df.head()
# 创建年龄区间特征
# pd.cut是按值的大小均匀切分，每组值区间大小相同，但样本数可能不一致
# pd.qcut是按照样本在值上的分布频率切分，每组样本数相同
train_df["AgeBand"] = pd.qcut(train_df["Age"],8)
train_df[["AgeBand","Survived"]].groupby(["AgeBand"],as_index = False).mean().sort_values(by="AgeBand",ascending=True)
# 把年龄按区间标准化为0到4
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 17, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 26), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 31), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 36.5), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 36.5) & (dataset['Age'] <= 45), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 7
train_df.head()
# 移除AgeBand特征
train_df = train_df.drop(["AgeBand"],axis=1)
combine = [train_df,test_df]
train_df.head()
# 创建家庭规模FamilySize组合特征
for dataset in combine:
    dataset["FamilySize"] = dataset["Parch"] + dataset["SibSp"] + 1
train_df[["FamilySize","Survived"]].groupby(["FamilySize"],as_index = False).mean().sort_values(by="FamilySize",ascending=True)

# 创建是否独自一人IsAlone特征
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1,"IsAlone"] = 1
train_df[["IsAlone","Survived"]].groupby(["IsAlone"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 移除Parch,Sibsp,FamilySize（暂且保留试试）
# 给字段赋值可以在combine中循环操作，删除字段不可以，需要对指定的df进行操作
train_df = train_df.drop(["Parch","SibSp"],axis=1)
test_df = test_df.drop(["Parch","SibSp"],axis=1)
combine = [train_df,test_df]
train_df.head()
# 创建年龄*级别Age*Pclass特征
# 这个有啥意义？
#for dataset in combine:
#    dataset["Age*Pclass"] = dataset["Age"] * dataset["Pclass"]
#train_df.loc[:,["Age*Pclass","Age","Pclass"]].head()
# 给Embarked补充空值
# 获取上船最多的港口
freq_port = train_df["Embarked"].dropna().mode()[0]
freq_port
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 把Embarked数字化
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map({"S":0,"C":1,"Q":2}).astype(int)
train_df.head()
# 去掉Embarked试试。。
#train_df = train_df.drop(["Embarked"],axis=1)
#test_df = test_df.drop(["Embarked"],axis=1)
#combine=[train_df,test_df]
#train_df.head()
# 给测试集中的Fare填充空值，使用中位数
test_df["Fare"].fillna(test_df["Fare"].dropna().median(),inplace=True)
test_df.info()
# 创建FareBand区间特征
train_df["FareBand"] = pd.qcut(train_df["Fare"],4)
train_df[["FareBand","Survived"]].groupby(["FareBand"],as_index=False).mean().sort_values(by="FareBand",ascending=True)
# 根据FareBand将Fare特征转换为序数值
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
test_df.head(10)
# 用seaborn的heatmap对特征之间的相关性进行可视化
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# 用seaborn的pairplot看各特征组合的样本分布
g = sns.pairplot(train_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
       u'FamilySize', u'Title', u'IsChildren', u'IsAlone', u'HasCabin',u'NameLength']], 
                 hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',
                 diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
# 有点浮夸，需要指点
X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn.model_selection import train_test_split

X_all = train_df.drop(['Survived'], axis=1)
y_all = train_df['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
# Random Forest
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
random_forest = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(random_forest, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)
clf = grid_obj.best_estimator_
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc_random_forest_split=accuracy_score(y_test, pred)
acc_random_forest_split


#random_forest.fit(X_train, Y_train)
#Y_pred_random_forest = random_forest.predict(X_test)
#random_forest.score(X_train, Y_train)
#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#acc_random_forest
from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)

Y_pred_random_forest_split = clf.predict(test_df.drop("PassengerId",axis=1))

#from sklearn.cross_validation import KFold

#def run_kfold(clf):
#    kf = KFold(891, n_folds=10)
#    outcomes = []
#    fold = 0
#    for train_index, test_index in kf:
#        fold += 1
#        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
#        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
#        clf.fit(X_train, y_train)
#        predictions = clf.predict(X_test)
#        accuracy = accuracy_score(y_test, predictions)
#        outcomes.append(accuracy)
#        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
#    mean_outcome = np.mean(outcomes)
#    print("Mean Accuracy: {0}".format(mean_outcome)) 

#run_kfold(clf)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, 
              acc_log, 
              acc_random_forest_split,
              #acc_random_forest,
              acc_gaussian, 
              acc_perceptron, 
              acc_sgd, 
              acc_linear_svc, 
              acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
import time
print(time.strftime('%Y%m%d%H%M',time.localtime(time.time())))
# 取最后更新的随机森林模型的预测数据进行提交

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_random_forest_split
        #"Survived": Y_pred_random_forest
    })
submission.to_csv('submission_random_forest_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_decision_tree
    })
submission.to_csv('submission_decision_tree'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_knn
    })
submission.to_csv('submission_knn_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_svc
    })
submission.to_csv('submission_svc_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)