import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from patsy import dmatrix # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
data.head()
data.shape
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
data.dtypes
surd = data.Survived.value_counts().plot(kind='bar')
surd.plot(kind='bar', label='Survived')
#plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar', color='m', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='bar', color='b', label='male')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Lowclass', color='y', alpha=0.6)
plt.show()
data.head()
data.shape
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)
X.shape
y.size
X.head()
y[:10]
model = LogisticRegression(C = 1e5)
model.fit(X, y)
answer = y
pred = model.predict(X)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
testdata = pd.read_csv('../input/test.csv')
testdata.head()
testdata = testdata.drop(['Ticket', 'Cabin'], axis = 1)
testdata = testdata.dropna()
testdata.head()
testdata.shape
testdata['Survived'] = 1
testdata.loc[np.isnan(testdata.Age), 'Age'] = np.mean(data['Age'])
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = testdata, return_type='dataframe')
Xtest.head()
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(testdata['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
solution.to_csv('./my_prediction.csv', index = False)

