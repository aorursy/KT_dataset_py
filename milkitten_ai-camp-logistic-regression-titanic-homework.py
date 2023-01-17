import numpy as np # 数组常用库

import pandas as pd # 读入csv常用库

from patsy import dmatrices # 可根据离散变量自动生成哑变量

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV # sk-learn库Logistic Regression模型

from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试

from sklearn import metrics # 生成各项测试指标库

import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')

data.head()
data = data.drop(['Ticket', 'Cabin', 'Name'], axis = 1)

data = data.dropna()

# for i, c in enumerate(data.columns):

#     if data[c].isnull().values.any():

#         if c == 'Sex': 

#             data[c] = data[c].fillna('female')

#         elif c == 'Embarked':

#             data[c] = data[c].fillna('S')   

#         else: 

#             data[c] = data[c].fillna(data[c].median())

     
len(data.index)
data.Survived.value_counts().plot.bar(alpha=0.5)

plt.show()
survived_f = data.Survived[data.Sex == 'female'].mean()

data.Survived[data.Sex == 'female'].value_counts().sort_index().plot.bar(alpha = 0.5)

plt.show()

print(str(survived_f*100 )+'% of female has survived')
survived_m = data.Survived[data.Sex == 'male'].mean()

data.Survived[data.Sex == 'male'].value_counts().plot.bar(alpha = 0.5)

plt.show()

print(str(survived_m )+'% of male has survived')
cross = pd.crosstab(data['Survived'], data['SibSp'])

print(cross)

cross.div(cross.sum(axis = 0), axis =1).plot.bar(stacked=True)

plt.show()
cross = pd.crosstab(data['Survived'], data['Parch'])

print(cross)

cross.div(cross.sum(axis = 0), axis =1).plot.bar(stacked=True)

plt.show()
cross = pd.crosstab(data['Survived'], data['Pclass'])

cross_p = cross.div(cross.sum(axis = 0), axis = 1)

print(cross_p)

cross_p.plot.bar(stacked = 'True')

plt.title('survived % of each class')

plt.show()



not_low = data.Survived[data.Pclass!=3].value_counts().sort_index()

not_low.plot.bar()

plt.title('survived (not in low class)')

plt.show()
low = data.Survived[data.Pclass==3].value_counts().sort_index()

low.plot.bar()

plt.title('survived (low class)')

plt.show()
y, X = dmatrices('Survived~C(Pclass)+C(Sex)+Age+Fare+C(Embarked)+SibSp+Parch', data, return_type = 'dataframe')

y = np.ravel(y)

X.head()

model = LogisticRegressionCV()
model.fit(X, y)
model.score(X,y)
dummy_model = 1 - data.Survived.mean()

dummy_model
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test = pd.read_csv('../input/test.csv')

test['Survived'] = 0

test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

print(test.shape)

for i, c in enumerate(test.columns):

    if(test[c].isnull().values.any()):

        test[c] = test[c].fillna(test[c].median())



dummy, Xtest = dmatrices('Survived~C(Pclass)+C(Sex)+Age+Fare+C(Embarked)+SibSp+Parch', test, return_type = 'dataframe')

print(Xtest.shape)



pred = model.predict(Xtest).astype(int)

print(pred.shape)
survived = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived' : pred})

survived.head()
survived.to_csv('./submission.csv', index = False)
df = pd.read_csv('./submission.csv')

df.head()