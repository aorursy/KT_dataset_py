import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库



from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
data = pd.read_csv('../input/train.csv')

data.head()
data = data.drop(['Ticket', 'Cabin'], axis=1)

data = data.dropna()
len(data.index)

data.shape
data.Survived.value_counts().plot(kind='bar')

plt.xlabel('Survived')

plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()

female.plot(kind='bar', label='Female')

plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()

male.plot(kind='bar', label='Male')

plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()

highclass.plot(kind='bar',label='Highclass', color='Green', alpha=0.5)

plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()

lowclass.plot(kind='bar',label='Lowclass', color='Purple', alpha=0.5)

plt.show()
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data=data, return_type='dataframe')

y = np.ravel(y)

X.head()
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
print(1 - y.mean())
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
model1 = LogisticRegression(C=10000)

model1.fit(X, y)

model1.score(X, y)
model2 = LogisticRegression(C=1000)

model2.fit(X, y)

model2.score(X, y)
model3 = LogisticRegression(C=100)

model3.fit(X, y)

model3.score(X, y)
model4 = LogisticRegression(C=10)

model4.fit(X, y)

model4.score(X, y)
Xtrain, Xvali, ytrain, yvali = train_test_split(X, y, test_size=0.25, random_state=31)

model_try = LogisticRegression(C=0.2, max_iter=50)

model_try.fit(Xtrain, ytrain)

pred = model_try.predict(Xvali)

print('train score: ' + str(model_try.score(Xtrain, ytrain)))

print('vali score: ' + str(metrics.accuracy_score(yvali, pred)))



cross_results = cross_val_score(model_try, X, y, scoring='accuracy', cv=5)

print(cross_results)

print('cross_scores_mean: ' + str(cross_results.mean()))


model_final = model_try

model_final.fit(X, y)

model_final.score(X, y)
test_data = pd.read_csv('../input/test.csv')



test_data['Survived'] = 1

test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])

ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')



pred = model_final.predict(Xtest).astype(int)

solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])

solution.to_csv('./my_prediction.csv', index=False)
