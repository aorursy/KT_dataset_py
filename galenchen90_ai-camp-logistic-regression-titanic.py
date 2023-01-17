import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

data.head()
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
data.head()
len(data.index)

data.index
data.shape
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()

lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()
model = LogisticRegression()


#dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived

# 这个的排序很重要：'Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)' 目测：需要呀变量的前面一个大写的C，最前面是要预测的东西 

y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')

y.head(10)
y = np.ravel(y) #把它变成array 方便模型fit
y[:10]
model.fit(X, y)

model.score(X, y)

#输出空模型的正确率：空模型预测所有人都未逃生
1 - y.mean()

pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

test_data.head()
test_data['Survived'] = 1  # 增加一列survived 方便后面操作～
test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])  #把 nan的变成平均年龄
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
pred.size
solution.to_csv('./my_prediction.csv', index = False)
