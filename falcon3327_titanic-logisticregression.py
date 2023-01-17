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
import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

# 查看数据行列

train.shape
# 检查NA值

train.isnull().sum()
# 筛选有用特征

train = train.drop(labels=["PassengerId","Name","Cabin","Ticket"],axis=1)

test = test.drop(labels=["Name","Cabin","Ticket"],axis=1)
# 去除有缺失值的行

train=train.dropna()
# 分类变量编码（字符串型转数字型）

train_dummy=pd.get_dummies(train[["Sex","Embarked"]])# 独热编码，和sklearn OneHot功能一样

train_conti = pd.DataFrame(train,columns=["Survived","Fare","Parch","SibSp","Age","Pclass"],index = train.index)

train = train_conti.join(train_dummy)# 数据拼接

test_dummy=pd.get_dummies(test[["Sex","Embarked"]])# 独热编码，和sklearn OneHot功能一样

test_conti = pd.DataFrame(test,columns=["PassengerId","Fare","Parch","SibSp","Age","Pclass"],index = test.index)

test = test_conti.join(test_dummy)# 数据拼接

test.info()

# 将测试集空数据补全

test["Age"] = test["Age"].fillna(np.mean(test["Age"]))

test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))

test.info()
x = train.iloc[:,1:]# 特征列

y = train.iloc[:,0]# 标签列

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)#数据集划分
# 标准化以保证数据的快速收敛

stdsc = StandardScaler()

x_train_conti_std=stdsc.fit_transform(x_train[["Age","SibSp","Parch","Fare"]])#拟合

x_test_conti_std=stdsc.fit_transform(x_test[["Age","SibSp","Parch","Fare"]])

#将ndarry转化成dataframe

x_train_conti_std=pd.DataFrame(x_train_conti_std,columns=["Age","SibSp","Parch","Fare"],index=x_train.index)

x_test_conti_std=pd.DataFrame(x_test_conti_std,columns=["Age","SibSp","Parch","Fare"],index=x_test.index)
# 使用逻辑回归建模

classifier = LogisticRegression(random_state=0)

classifier.fit(x_train,y_train)# 模型训练
# 将模型应用于测试集并查看混淆矩阵

y_pred = classifier.predict(x_test)

# 打印混淆矩阵

confusion_matrix = confusion_matrix(y_test,y_pred)

print(confusion_matrix )
# 打印准确率

print("Accuracy :{:.2f}".format(classifier.score(x_test,y_test)))
print(test)

predict = classifier.predict(test.iloc[:,1:])

submission = pd.DataFrame({'PassengerId':test["PassengerId"],'Survived':predict})# 以字典的形式建立dataframe

submission.to_csv("submmision_csv",index = False)# 转换为CSV文件

pd.read_csv("submmision_csv")