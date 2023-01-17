# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # 基于matplotlib的图形可视化包
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost.sklearn import XGBClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
# Part1:数据分析
# 在此鸣谢aicanghai_hai的博客，科学性地教会了我怎么样进行这类数据的分析与转化

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# 查看表格表头类型
display(train.head(n=0), test.head(n=0))
# 查看数据整体信息
train.info()
test.info()
# 分析后可知，“carbin”数据项在训练集和测试集均有极大程度的缺失，决定删去此数据项，以免造成干扰
#   但其实理论上来说可以通过票的编号等齐全的信息来算出“carbin”
# 此外，那些仅有部分损失的数据项, 决定以NULL或者均值代替补全，以保留那些更加有用的信息
# 让我给不同数据项对生存率的影响力递减排序的话：Age，sex, Pcalss(ticket classs), parch(num of parensts/..)
#   sibsp(num of siblings/..), (name, ticket, fare , embarked, passengerID)

# 使用seaborn库进行可视化分析
#   sns.countplot(x=None, y=None, hue=None, data=None, order=None, 
#     hue_order=None, orient=None, color=None, palette=None, saturation=0.75, 
#       dodge=True, ax=None, **kwargs)
plt.figure(0)
sns.countplot(x='Pclass', hue='Survived', data=train)
# 很容易就可以发现只有class 1(天龙人)存活率高于死亡率，只有class 3(首陀罗)死亡率远高于存活率
plt.figure(1)
sns.countplot(x='Sex', hue='Survived', data=train)
# 男士生存率远低于死亡率，女士生存率远高于死亡率，重要因素
plt.figure(2)
sns.countplot(x='Parch', hue='Survived', data=train)
# 有且就仅有一个parch可以显著增加存活率，有两个会部分提升
train['Parch'] = train['Parch'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')
# 泛化区间
plt.figure(3)
sns.countplot(x='SibSp', hue='Survived', data=train)
train['SibSp'] = train['SibSp'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')
# 和SibSp一致，总体上看就是有且仅有一名家属可以大大提高存活率，而两名也会有部分提升，继续增加变化不大
# 小提琴图：
#   seaborn.violinplot(x=None, y=None, hue=None, data=None, order=None, 
#     hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, 
#       width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, 
#         color=None, palette=None, saturation=0.75, ax=None, **kwargs)
plt.figure(4)
sns.violinplot(x='Survived', y='Age', data=train, split=True, scale="count")
# 儿童的存活比最高，青年最低，在不同的年龄段有清晰的界限
#    然后这类自变量很多，接近连续的分布，可以选择将其划分成几个主要的大类：儿童，青年，中年，老年
#       0-17：净存活率++ ；17-30：净存活率-- ；30-50：净存活率：- ；50+: 净存活率：+
train['Age'] = train['Age'].map(lambda x: 'kids' if x<17 else 'youth' if x<30 else 'mid ages' if x<50 else 'olds')
#         通过上述语句实现了区间泛化
plt.figure(5)
sns.violinplot(x='Survived', y='Fare', data=train, scale='count')
# 虽然很明显高消费人群的存活率更高，但具体数字难以分辨，进行对数转换以使得因变量的变化不那么剧烈, 顺便补全了它
train['Fare'] = train['Fare'].map(lambda x: np.log(x+1))
plt.figure(6)
sns.violinplot(x='Survived', y='Fare', data=train, scale='count')
# 做同样的泛化区间操作
train['Fare'] = train['Fare'].map(lambda x: 'poor' if x<2.5 else 'wealthy')
# 'cabin'数据缺省太多，决定处理掉它
#   pandas.DataFrame.dropna : pandas包下DataFrame中的一个删除缺失值的用法
#     DataFrame.dropna(axis=0,how='any',thresh=None,subset=None,inplace=False)
#       axis(0删行/1删列) how(any任何NA存在就删除所在行或者列/all行或者列全为NA才能删除)
#           inplace(True在原表上修改/False不在原表上修改)
plt.figure(7)
sns.countplot( x='Embarked', hue='Survived', data=train)
# 还是有点影响的，但先不用了
train.dropna(axis=1, inplace=True)
train.info()


# Part 2:数据处理
# 可删除的数据有'cabin'and'Embarked','name','ticket'
# 将数据分成标记与特征:
labels = train['Survived']
features = train.drop(['PassengerId', 'Survived','Name','Ticket'], axis=1)
features.info()

# 将数据进行one-hot编码
# pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
#   可将数据进行one-hot编码，将str信息变成01信息
features = pd.get_dummies(features)
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# 处理测试集
# 补齐Age和Fare
# DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
# 进行区间泛化
test['Age'] = test['Age'].map(lambda x: 'kids' if x<17 else 'youth' if x<30 else 'mid ages' if x<50 else 'olds')
test['Fare']=test['Fare'].map(lambda x:np.log(x+1))
test['Fare'] = test['Fare'].map(lambda x: 'poor' if x<2.5 else 'wealthy')
test['Parch'] = test['Parch'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')
test['SibSp'] = test['SibSp'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')
ID = test['PassengerId']
# 删去不需要的项
test=test.drop(['PassengerId', 'Cabin','Embarked','Name','Ticket'],axis=1)
test.info()
test=pd.get_dummies(test)
encoded = list(test.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))





# Part 3:机器学习算法

# SVC : support vector classification
# method 1:
#定义通用函数框架
def fit_model(alg,parameters):
    X=features
    y=labels  #由于数据较少，使用全部数据进行网格搜索
    scorer=make_scorer(roc_auc_score)  #使用roc_auc_score作为评分标准
    grid = GridSearchCV(alg,parameters,scoring=scorer,cv=5)  #使用网格搜索，自动调参
    start=time()  #计时
    grid=grid.fit(X,y)  #模型训练
    end=time()
    t=round(end-start,3)
    print (grid.best_params_)  #输出最佳参数
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    return grid #返回训练好的模型

alg_svc = SVC(probability=True,random_state=29)
parameters_svc = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25]}

clf_svc = fit_model(alg_svc, parameters_svc)


# 测试集标签验证并修改文件
def save(clf,i):
    pred=clf.predict(test)
    sub=pd.DataFrame({ 'PassengerId': ID, 'Survived': pred })
    sub.to_csv("res_tan_{}.csv".format(i), index=False)

save(clf_svc, 1)






        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
