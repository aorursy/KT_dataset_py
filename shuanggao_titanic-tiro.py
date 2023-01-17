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
import pandas as pd



titanic = pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')

print(titanic)

#describe()只能统计数据中的数值量

print(titanic.describe())
#由describe()可知，Age的count只有714个是有缺失值的，因此用Age的中位数对此进行填充

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

print(titanic.describe())



#对数据中我们认为重要的非数值型的量，进行数值表示，方便机器进行学习。例如：Sex

#先查看Sex中有几类数据

print(titanic['Sex'].unique())

# loc[行标签，列标签] 按照行列标签进行切片，

titanic.loc[titanic['Sex']== 'male' ,'Sex'] = 0

titanic.loc[titanic['Sex']== 'female' ,'Sex'] = 1



#将上船地点进行数值化表示

print(titanic['Embarked'].unique())

titanic['Embarked'] = titanic['Embarked'].fillna('S')

#数值化表示

titanic.loc[titanic['Embarked']=='S','Embarked'] = 0

titanic.loc[titanic['Embarked']=='C','Embarked'] = 1

titanic.loc[titanic['Embarked']=='Q','Embarked'] = 3



print('处理后的数据：\n',titanic.head())
#可能家庭成员的数量对是否获救有关，则对该特征进行提取

titanic['FamilySize'] = titanic['SibSp']+titanic['Parch']

#可能名字长度对是否获救也有某种潜在的关联

titanic['NameLength']=titanic['Name'].apply(lambda x:len(x))



#使用正则表达式来对名字进行提取

import re



def get_title(name):

    #名字总是由大小写字母组成，并以点号（.）结束

    #在name中寻找title

    title_search = re.search('([A-Za-z]+)\.',name)

    #如果存在

    if title_search:

        # 数据集的名字如Todoroff, Mr. Lalio 我们要找出其中的Mr 则返回group(1)

        return title_search.group(1)

    return ''



#获取所有title（如Mr，Miss等） 并对其进行统计个数

titles = titanic['Name'].apply(get_title)

print(titles)

print(pd.value_counts(titles))



#将字符标签（Mr，Mrs,Miss等）数值化



titles_mapping = { 

    "Mr": 1,

    "Miss": 2,

    "Mrs": 3,

    "Master": 4,

    "Dr": 5,

    "Rev": 6,

    "Major": 7,

    "Col": 7,

    "Mlle": 8,

    "Mme": 8,

    "Don": 9,

    "Lady": 10,

    "Countess": 10,

    "Jonkheer": 10,

    "Sir": 9,

    "Capt": 7,

    "Ms": 2

}



for k,v in titles_mapping.items():

    titles[k==titles]=v

print(pd.value_counts(titles))



#给原数据添加title特征

titanic['titles']=titles

print('新增特征后的train数据集：\n',titanic.head())

import numpy as np

from sklearn.feature_selection import SelectKBest,f_classif #选择最好特征

import matplotlib.pyplot as plt



features = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked','FamilySize','NameLength','titles']

print('提取特征后的数据：\n',titanic[features].head())



#执行特征选择

selector = SelectKBest(f_classif,k=5) 

selector.fit(titanic[features],titanic['Survived']) 



#获取每个特征的得分

scores = -np.log10(selector.pvalues_)

print('每个特征的得分列表：\n',scores)



#绘制得分图，看哪些特征对我们来说是重要的

plt.bar(range(len(features)),scores)

plt.xticks(range(len(features)),features,rotation = 'vertical') #用features中的特征名来代替range(len(features))

plt.show()
from sklearn.externals import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



#我们所需要的特征量features

features = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']



X = titanic.loc[:,features] #iloc[]切片按行列标签

Y = titanic.loc[:,'Survived']

#print('数据特征X：\n',X)

#print('数据的标签Y：\n',Y)



#将X 和Y split为训练集train 和测试集test

#split X and Y into train and test

seed = 7

test_size = 0.33 #将数据集中33%的数据用来测试

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)

print('X_train_shape:',X_train.shape)

print('X_test_shape:',X_test.shape)

print('Y_train_shape:',Y_train.shape)

print('Y_test_shape:',Y_test.shape)

#训练XGBClassifier分类模型

model = XGBClassifier().fit(X_train,Y_train)



#用测试集对分类的结果进行预测，看分类模型的准确率

Y_pred = model.predict(X_test)

print('Y_pred:\n',Y_pred)

print('Y_test:\n',Y_test.values)

predictions = [round(value) for value in Y_pred]

#accuracy准确率为测试集标签的预测与实际测试集标签之间的正确率，也可以用来看我数据的分类是否合理

accuracy = accuracy_score(Y_test,predictions)

print('Accuracy:%.2f%%'%(accuracy*100))
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

import numpy as np



#我们所需要的特征量features

features = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']

# print(titanic[features])



model = LinearRegression()

kf = KFold(n_splits = 10,random_state = 1,shuffle=True)



predictions = []



for train,test in kf.split(titanic):

    train_features = titanic[features].iloc[train,:]

    train_label = titanic['Survived'].iloc[train]

    model.fit(train_features,train_label)

    test_predictions = model.predict(titanic[features].iloc[test,:])

    predictions.append(test_predictions)

# print(predictions)

predictions = np.concatenate(predictions,axis=0)

# print(predictions)

#predictions中存放的是test样本中预测的概率

predictions[predictions>0.5]=1

predictions[predictions<=0.5]=0

print('数据的原标签（1表示获救）：\n',titanic['Survived'].to_list())

print('预测的标签（1表示获救）：\n',predictions)

accurary = sum(predictions[predictions==titanic['Survived']])/len(predictions)

print('模型在训练集上的分类的准确度：%.2f%%'%(accuracy*100))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



features = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']

model = RandomForestClassifier(random_state=1,n_estimators=25,min_samples_split=4,min_samples_leaf=2)

# n_estimators=10 随机森林里要构建树的个数  min_samples_split=2 数据最小切分个数  min_samples_leaf=1 叶子结点的最小个数

kf = KFold(n_splits=3,random_state=1,shuffle=False)

scores = cross_val_score(model,titanic[features],titanic['Survived'],cv=kf)

print('模型在训练集上的分类准确率：%.2f%%'%(scores.mean()*100))
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



features = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']

X = titanic.loc[:,features]  #iloc[]切片按行列标签

Y = titanic.loc[:,'Survived']

#print('数据特征X：\n',X)

#print('数据的标签Y：\n',Y)



seed = 7

test_size = 0.33 #将数据集中33%的数据用来测试

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)

model = KNN().fit(X_train,Y_train)

y_pred = model.predict(X_test)



print('Y_pred:\n',Y_pred)

print('Y_test:\n',Y_test.values)

predictions = [round(value) for value in Y_pred]

#accuracy准确率为测试集标签的预测与实际测试集标签之间的正确率，也可以用来看我数据的分类是否合理

accuracy = accuracy_score(Y_test,predictions)

print('Accuracy:%.2f%%'%(accuracy*100))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

import numpy as np



#组合我们要集成的算法，将多种算法集成为一个模型

features = ['Pclass','Age','Sex','Fare','Embarked','FamilySize','NameLength','titles']

# features = ['Pclass','Age','Sex','Fare','Embarked','FamilySize','titles']

model = [

    [GradientBoostingClassifier(random_state=1,n_estimators=1000,min_samples_split=4,min_samples_leaf=2,max_depth=4),features],

    [LogisticRegression(random_state = 1,solver = 'liblinear'),features],

]



#初始化和交叉验证

kf = KFold(n_splits=10,shuffle= False,random_state=1)



predictions=[]  

for train,test in kf.split(titanic):

    train_lable = titanic['Survived'].iloc[train]

    full_test_predictions = [] #存放的是一折情况下每个模型的预测的概率

    #对集成算法里的每个算法都用每折数据进行预测

    for model_i,feature in model:

        #用train数据对每个模型进行训练

        model_i.fit(titanic[features].iloc[train,:],train_lable)

        #避免数据类型的错误，都转换成float型

        test_predictions = model_i.predict_proba(titanic[features].iloc[test,:].astype(float))[:,1]

#         test_predictions = model_i.predict(titanic[features].iloc[test,:])

        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[0]+full_test_predictions[1])/2

    test_predictions[test_predictions > 0.5] = 1

    test_predictions[test_predictions <= 0.5] = 0

    predictions.append(test_predictions)



#将所有折test数据的预测结果放在一个数组中

predictions = np.concatenate(predictions,axis=0)

print('用交叉验证中的每一折test预测的标签：\n',predictions)



print('原始数据的标签：\n',titanic['Survived'].to_list())



#与训练集的数据比较得出正确率

accuracy = sum(predictions == titanic['Survived'])/len(predictions)

print('在训练集上，集成算法的准确性：%.2F%%'%(accuracy*100))
import pandas as pd

import re



titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv',index_col = 'PassengerId')

# print('测试数据集：\n',titanic_test)

# print(titanic_test.describe())



#发现Age和Fare 有缺失值，进行填补

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())

# print(titanic_test.describe())



#对字符型特征数值化

# print(titanic['Sex'].unique())

titanic_test.loc[titanic_test['Sex']=='male','Sex'] = 0

titanic_test.loc[titanic_test['Sex']== 'female' ,'Sex'] = 1



#对字符型特征数值化

# print(titanic_test['Embarked'].unique())

titanic_test.loc[titanic_test['Embarked']=='S','Embarked'] = 0

titanic_test.loc[titanic_test['Embarked']=='C','Embarked'] = 1

titanic_test.loc[titanic_test['Embarked']=='Q','Embarked'] = 3

# print(titanic_test.head())



#对test数据集的其他特征进行处理，添加其他提取的特征



#可能家庭成员的数量对是否获救有关，则对该特征进行提取

titanic_test['FamilySize'] = titanic_test['SibSp']+titanic_test['Parch']

#可能名字长度对是否获救也有某种潜在的关联

titanic_test['NameLength']=titanic_test['Name'].apply(lambda x:len(x))



#使用正则表达式来对名字进行提取

def get_title(name):

    #名字总是由大小写字母组成，并以点号（.）结束

    #在name中寻找title

    title_search = re.search('([A-Za-z]+)\.',name)

    #如果存在

    if title_search:

        # 数据集的名字如Todoroff, Mr. Lalio 我们要找出其中的Mr 则返回group(1)

        return title_search.group(1)

    return ''



#获取所有title（如Mr，Miss等） 并对其进行统计个数

titles = titanic_test['Name'].apply(get_title)

# print(titles)

# print(pd.value_counts(titles))



#将字符标签（Mr，Mrs,Miss等）数值化



titles_mapping = { 

    "Mr": 1,

    "Miss": 2,

    "Mrs": 3,

    "Master": 4,

    "Dr": 5,

    "Rev": 6,

    "Major": 7,

    "Col": 7,

    "Mlle": 8,

    "Mme": 8,

    "Dona": 9,

    "Lady": 10,

    "Countess": 10,

    "Jonkheer": 10,

    "Sir": 9,

    "Capt": 7,

    "Ms": 2

}



for k,v in titles_mapping.items():

    titles[k==titles]=v

# print(pd.value_counts(titles))

titanic_test['titles']=titles

print(titanic_test.head())



#使用集成算法预测

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import numpy as np



#组合我们要集成的算法，将多种算法集成为一个模型

features = ['Pclass','Age','Sex','Fare','Embarked','FamilySize','titles']

model = [

    [GradientBoostingClassifier(random_state=1,n_estimators=1000,min_samples_split=4,min_samples_leaf=2,max_depth=4),features],

    [LogisticRegression(random_state = 1,solver = 'liblinear'),features],

]



full_predictions = []

for model_i,features in model:

    # 用整个训练集对模型进行训练.

    model_i.fit(titanic[features],titanic['Survived'])

    # 使用测试数据集进行预测。我们必须将所有列都转换为浮点数以避免错误.

    predictions = model_i.predict_proba(titanic_test[features].astype(float))[:, 1]

    predictions[predictions <= .5] = 0

    predictions[predictions > .5] = 1

    full_predictions.append(predictions)

print(predictions)



#将预测结果保存为CSV文件

test = pd.DataFrame(predictions,index = titanic_test.index,dtype = 'int',columns=['Survived'])

print(test)

test.to_csv('Predictions_Survived.csv')

print('Done')
