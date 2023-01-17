import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices,dmatrix # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
data = pd.read_csv ("../input/train.csv", sep=",")
testset= pd.read_csv ("../input/test.csv", sep=",")
testset['Survived']=0
#做成全数据，方便统一来进行数据处理
all_data=pd.concat([data,testset])

all_data = all_data.drop(['Ticket', 'Cabin'], axis = 1)
#data = data.dropna()
#all_data['name_title'] = all_data['Name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
#all_data['name_len']=all_data.Name.apply(lambda x:len(x))
all_data.loc[:,'Embarked']=all_data['Embarked'].fillna('S')
all_data['Sex']=all_data['Sex'].fillna('male')
#all_data.loc[:,'Age']=all_data['Age'].fillna(all_data['Age'].median())
all_data.loc[all_data['Sex']=='male','Age']=all_data.loc[all_data['Sex']=='male','Age'].fillna(all_data.loc[all_data['Sex']=='male','Age'].median())
all_data.loc[all_data['Sex']=='female','Age']=all_data.loc[all_data['Sex']=='female','Age'].fillna(all_data.loc[all_data['Sex']=='female','Age'].median())
all_data.head()
data['Survived'].value_counts().plot(kind='bar')

plt.show()
data.loc[data['Sex']=='female','Survived'].value_counts().sort_index().plot(kind='bar')

plt.show()
data.loc[data['Sex']=='male','Survived'].value_counts().plot(kind='bar')

plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='lowclass', color='blue', alpha=0.6)
plt.show()
all_data.isnull().any()
all_data.Age.plot(kind='hist')
#这里需要手动输入q,因为28岁的人太多了，重叠的区间太多
all_data['age_level']=pd.qcut(all_data['Age'],q=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.55,0.65,0.7,0.75,0.8,0.9,1])
#data=all_data[:len(data)]
#testset=all_data[len(data):]
#print(all_data.shape,data.shape,testset.shape)
y,X=dmatrices('Survived~C(Pclass)+C(Sex)+C(Embarked)+C(age_level)+C(SibSp)', all_data, return_type='dataframe')
y=np.ravel(y)[:len(data)]
Xtest=X[len(data):]
X=X[:len(data)]

print(Xtest.shape,X.shape)
model = LogisticRegression(C=10,penalty='l2')
model.fit(X, y)
pred=model.predict(X)
metrics.accuracy_score(y,pred)

metrics.accuracy_score(y,[0]*len(y))
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
print(cross_val_score(model, X, y, scoring='accuracy', cv=10).mean())
#用GridSearchCV 寻找最好的C和penalty参数
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(LogisticRegression(), scoring="accuracy", cv=5, verbose=1,
                  param_grid={"C": [0.001,0.01,0.1,1,10,100,1000,10000],"penalty":["l1","l2"]}, )
gs.fit(X, y)
print("best params",gs.best_params_,"best scores:",gs.best_score_)

model = LogisticRegression(C=10,penalty='l1')
model.fit(X, y)
prediction=model.predict(Xtest)
prediction=prediction.astype(int)
df=pd.DataFrame({"PassengerId":testset.PassengerId,"Survived":prediction})
df.to_csv("titanic_logistic_regression.csv",header=True,index=False)
