import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
len(data.index)
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar', color='blue', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='bar',label='Male', color='red')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)
model = LogisticRegression(C = 0.1)
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv('../input/test.csv')
test_data['Survived'] = 1   # add this column so we can use it in Logistic Regression model
test_data['Age'].fillna(np.mean(data['Age']))      # fill missing data with mean of Age
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
solution.to_csv('../my_prediction.csv', index = False)    # write result to data file

