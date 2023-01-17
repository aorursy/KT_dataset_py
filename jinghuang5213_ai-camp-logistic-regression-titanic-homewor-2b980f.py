import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
len(data.index)
data.Survived.value_counts().plot(kind='bar')

female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar')
female = data.Survived[data.Sex == 'male'].value_counts()
female.plot(kind='bar') 
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar')
plt.show()
highclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
highclass.plot(kind='bar')
plt.show()
y, X = dmatrices('Survived~C(Pclass) + C(Sex) + Age + C(Embarked)', data=data, return_type='dataframe')
y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X,y)
1-y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test['Survived'] = 0
test.loc[np.isnan(test.Age), 'Age'] = np.mean(data['Age'])
ytest, Xtest=dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', data=test,return_type='dataframe')
pred=model.predict(Xtest).astype(int)
sol=pd.DataFrame(list(zip(test['PassengerId'],pred)), columns=['PassengerId', 'Survived'])
sol.to_csv('./my_prediction.csv', index = False)