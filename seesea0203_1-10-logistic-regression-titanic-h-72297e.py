import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
len(data.index)
import seaborn as sns
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
data.Survived[data.Sex == 'female'].value_counts().sort_index().plot(kind='barh', color='b', label='Female')

plt.show()
data.Survived[data.Sex == 'male'].value_counts().sort_index().plot(kind='barh', color='r', label='Male')
plt.show()
sns.countplot(data.Survived, hue=data.Sex)
plt.xlabel('Survived')
plt.show()
survived_male = data.Survived[data.Sex=='male'].value_counts().sort_index()
survived_female = data.Survived[data.Sex=='female'].value_counts().sort_index()
pd.DataFrame({'Male': survived_male, 'Female': survived_female}).plot(kind='bar', stacked=True)
plt.title('Survived-Sex')
plt.legend(loc='best')
plt.xlabel('Dead or Survived')
plt.ylabel('Count')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar', label='Lowclass', color='b', alpha=0.5)
plt.show()
sns.countplot(data.Pclass, hue=data.Survived)
plt.show()
y, X = dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', data=data, return_type='dataframe')
y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X,y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test = pd.read_csv('../input/test.csv')
test.info()
test.Age = test.Age.fillna(data.Age.mean())
ytest, Xtest = dmatrices('Parch~C(Pclass)+C(Sex)+C(Embarked)+Age', data=test, return_type='dataframe')
# 训练和测试中用dmatrices进行哑变量转变时，如果features的顺序不同应该不会对预测结果产生影响吧？
# 这里Parch~Age+C(Embarked)+C(Sex)+C(Pclass)应该也是一样的对么？
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test.PassengerId, pred)), columns=['PassengerID', 'Survived'])
solution.to_csv('./submission.csv', index=False)