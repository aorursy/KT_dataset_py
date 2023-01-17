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
data['Survived'].value_counts().sort_index().plot(kind='bar')

plt.xlabel('Survived')

plt.ylabel('Count')

plt.show()
data.Survived[data['Sex'] == 'female'].value_counts().sort_index().plot(kind='barh', color='blue')

plt.xlabel('Survived')

plt.ylabel('Count')

plt.title('Female Survival')

plt.show()
data.Survived[data['Sex'] == 'male'].value_counts().sort_index().plot(kind='barh', color='red')

plt.xlabel('Survived')

plt.ylabel('Count')

plt.title('Male Survival')

plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()

highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)

plt.legend()

plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()

lowclass.plot(kind='bar', color='green', alpha=0.7, label='Lowclass')

plt.legend()

plt.show()
data.columns

y, X = dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', data, return_type='dataframe')



y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test = pd.read_csv('../input/test.csv')



test['Survived'] = 1

# fill Nan

test.loc[np.isnan(test['Age']),'Age'] = np.mean(test['Age'])



# convert to dummy variable

ytest, Xtest = dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', \

                         test[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked']], \

                         return_type='dataframe')



# predict on test

pred_test = model.predict(Xtest)



pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':pred_test}).to_csv('submission.csv')