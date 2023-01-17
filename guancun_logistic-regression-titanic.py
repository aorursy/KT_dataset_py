import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

data.head(5)



print(data.shape)

print(test_data.shape)
data
data = data.drop(['Ticket', 'Cabin'], axis = 1)

data.shape
data = data.dropna()

data.shape
data.Survived.value_counts().plot(kind='bar')

plt.show()
data.Survived[data.Sex=='male'].value_counts().sort_index().plot(kind='barh', color='blue', label='Female')

plt.show()
data.Survived[data.Sex=='female'].value_counts().sort_index().plot(kind='barh', color='red', label='Female')

plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()

highclass.plot(kind='bar', color='red', label='Highclass', alpha = 0.5)

plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()

lowclass.plot(kind='bar', color='blue', label='Lowclass', alpha = 0.5)

plt.show()
y, X = dmatrices('Survived ~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type = 'dataframe')

y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X,y)
1 - sum(y) / len(y)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
data['Age_group'] = 100

data.loc[data['Age'] <= 10, 'Age_group'] = 0

data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'Age_group'] = 1

data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'Age_group'] = 2

data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age_group'] = 3

data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age_group'] = 4

data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age_group'] = 5

data.loc[data['Age'] > 60, 'Age_group'] = 6
data['Survived'].groupby(data['Age_group']).mean()
y2, X2 = dmatrices('Survived ~ C(Pclass) + C(Sex) + C(Age_group) + C(Embarked)', data = data, return_type = 'dataframe')

y2 = np.ravel(y2)
model2 = LogisticRegression()
model2.fit(X2, y2)
model2.score(X2,y2)
pd.DataFrame(list(zip(X2.columns, np.transpose(model2.coef_))))
data['name_title'] = data['Name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
data['name_title'].value_counts()
data['Survived'].groupby(data['name_title']).mean()
y3, X3 = dmatrices('Survived ~ C(Pclass) + C(Sex) + C(Age_group) + C(Embarked) + C(name_title)', data = data, return_type = 'dataframe')

y3 = np.ravel(y3)
model3 = LogisticRegression()
model3.fit(X3, y3)
model3.score(X3,y3)
pd.DataFrame(list(zip(X3.columns, np.transpose(model3.coef_))))
test_data['Survived'] = 1

test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])
test_data['Age_group'] = 100

test_data.loc[test_data['Age'] <= 10, 'Age_group'] = 0

test_data.loc[(test_data['Age'] > 10) & (test_data['Age'] <= 20), 'Age_group'] = 1

test_data.loc[(test_data['Age'] > 20) & (test_data['Age'] <= 30), 'Age_group'] = 2

test_data.loc[(test_data['Age'] > 30) & (test_data['Age'] <= 40), 'Age_group'] = 3

test_data.loc[(test_data['Age'] > 40) & (test_data['Age'] <= 50), 'Age_group'] = 4

test_data.loc[(test_data['Age'] > 50) & (test_data['Age'] <= 60), 'Age_group'] = 5

test_data.loc[test_data['Age'] > 60, 'Age_group'] = 6
test_data['name_title'] = test_data['Name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
test_data
data['name_title'].value_counts()
test_data['name_title'].value_counts()
ytest, Xtest = dmatrices('Survived ~ C(Pclass) + C(Sex) + C(Age_group) + C(Embarked)', data = test_data, return_type = 'dataframe')
pred = model2.predict(Xtest).astype(int)

pred
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns = ['PassengerID', 'Survived'])

solution
solution.to_csv('../input/my_prediction.csv', index = False)
ytest, Xtest = dmatrices('Survived ~ C(Pclass) + C(Sex) + C(Age_group) + C(Embarked) + C(name_title)', data = test_data, return_type = 'dataframe')
pred2 = model3.predict(Xtest).astype(int)

pred2