import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv("../input/train.csv")
data.head()
data = data.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
data = data.dropna() 
data.head()
len(data.index)
data['Survived'].value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data['Survived'][data['Sex'] == 'female'].value_counts().sort_index() #sort_index()--Sort object by labels
female.plot(kind = 'barh', color = 'blue', title = 'Female')
plt.xlabel('Survived')
plt.show()
male = data['Survived'][data['Sex'] == 'male'].value_counts().sort_index() 
male.plot(kind = 'barh', color = 'red', title = 'Male')
plt.xlabel('Survived')
plt.show()
male = data['Survived'][data['Sex'] == 'male'].value_counts().sort_index() 
male.plot(kind = 'pie', title = 'Male')
plt.show()
highclass = data['Survived'][data.Pclass != 3].value_counts().sort_index() 
highclass.plot(kind = 'bar', color = 'blue', title = 'HighClass', alpha = 0.6)
plt.xlabel('Survived')
plt.show()
lowclass = data['Survived'][data.Pclass == 3].value_counts().sort_index() 
lowclass.plot(kind = 'bar', color = 'red', title = 'LowClass', alpha = 0.6)
plt.xlabel('Survived')
plt.show()
y, X = dmatrices('Survived~C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
X.head()
y = np.ravel(y)
y
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv("../input/test.csv")
test_data.head()
test_data['Survived'] = 1 #也可以写成test_data.Survived = 1
test_data.head()
test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])
test_data.Age.isnull().values.any() #检测是否全部补齐数据
ytest, Xtest = dmatrices('Survived~C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
Xtest.head()
pred = model.predict(Xtest).astype(int)
result = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns = ['PassengerID', 'Survived'])
result.to_csv('./my_prediction.csv', index = False)
