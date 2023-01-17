import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv('../input/train.csv')
print(data.shape)
print(data.columns)
data.head(5)
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
data.shape
len(data.index)
data['Survived'].value_counts().plot(kind = 'bar')
plt.show()
data['Survived'].value_counts(normalize = True)
data[data.Sex == 'female'].Survived.value_counts().plot(kind = 'bar')

data[data.Sex == 'female'].Survived.value_counts(normalize = True)
data[data.Sex == 'male'].Survived.value_counts().plot(kind = 'bar')

data[data.Sex == 'male'].Survived.value_counts(normalize = True)
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
import seaborn as sns

plt.title('3rd class sruvival distribution')
sns.countplot(data[data.Pclass == 3].Survived)
plt.show()
y, X = dmatrices('Survived ~ C(Pclass) + C(Sex) + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)
X.head()
model = LogisticRegression()
model.fit(X, y)
model.score(X,y)
1-y.sum()/len(y)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
from patsy import dmatrix
data_test = pd.read_csv('../input/test.csv')
Xtest = dmatrix('C(Pclass) + C(Sex) + C(Embarked)', data = data_test, return_type='dataframe')
pred = model.predict(Xtest)
result = pd.DataFrame(list(zip(data_test['PassengerId'], pred)), columns = ['PassengerId', 'Survived'])
result.to_csv('./my_prediction.csv', index = False)