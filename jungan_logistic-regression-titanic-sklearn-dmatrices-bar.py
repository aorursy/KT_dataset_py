import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv("../input/train.csv")
data = data.drop(['Ticket','Cabin'], axis = 1)
data = data.dropna() # 只要某一行，里面有一列的值为NA, 就删除改一整行
len(data.index)
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='Highclass', color='Blue', alpha=0.6)
plt.show()
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type='dataframe')
y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)
1 - y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv('../input/test.csv')
#test_data.head(): there is no label column “Survived”, for dmatrices purpose, you need to manually add dummy column
test_data['Survived'] = 1
test_data.head()
#test_data.isnull().sum(), will list NAN value counts for each column

#fill nan
# https://www.kaggle.com/jungan/ai-camp-logistic-regression-homework-025178?scriptVersionId=5790279
#data_copy=data.copy(deep=True)
#data_copy.loc[:, 'Age'] = data_copy.Age.fillna(data_copy.Age.median())

test_data.loc[np.isnan(test_data.Age), 'Age'] = np.mean(data['Age'])
# double check
#test_data.isnull().sum()
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')

pred = model.predict(Xtest).astype(int)
# columns=['PassengerID', 'Survived'] just for adding new titles for columns 
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
#solution.head()
solution.to_csv('./my_prediction.csv', index = False)
