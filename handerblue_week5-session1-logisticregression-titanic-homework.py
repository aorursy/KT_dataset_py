import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv("../input/train.csv")

data.head()
data = data.drop(['Ticket', 'Cabin'], axis = 1)

data = data.dropna()

data['name_length'] = [len(name) for name in data['Name']]

data.head()
len(data.index)
Survived_count = data["Survived"].value_counts()

Survived_count.plot(kind='bar')

plt.title("Survived Count")

plt.show()
Survived_gender = pd.crosstab(data['Survived'], data['Sex'])

Survived_gender['female'].plot(kind = 'barh', color ='red')

plt.title("Survived Count by Female")

plt.show()
Survived_gender['male'].plot(kind = 'barh', color = 'blue')

plt.title("Survived Count by Male")

plt.show()
Survived_Pclass = pd.crosstab(data['Survived'], data[data['Pclass'] != 3]['Pclass'])

Survived_Pclass.plot(kind = 'bar', stacked = True)

plt.title("Survived by High Classes")

plt.show()
Survived_Pclass = pd.crosstab(data['Survived'], data[data['Pclass'] == 3]['Pclass'])

Survived_Pclass.plot(kind = 'bar', color ='g')

plt.title("Survived of Low Class")

plt.show()
data.head()
y, X = dmatrices("Survived~ C(Pclass) + C(Sex) + Age + C(Embarked) + name_length + Fare", data, return_type = 'dataframe')

y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv("../input/test.csv")

test_data['Survived'] = 1

test_data = test_data.drop(['Ticket', 'Cabin'], axis = 1)

test_data = test_data.dropna()

test_data['name_length'] = [len(name) for name in test_data['Name']]

testy, testX = dmatrices("Survived~ C(Pclass) + C(Sex) + Age + C(Embarked) + name_length + Fare", test_data, return_type = 'dataframe')



pred = model.predict(testX)

solution = pd.DataFrame(list(zip(test_data["PassengerId"], pred)), columns = ["PassengerId", "Survived"])
solution.to_csv('./my_prediction.csv', index = False)