import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/titanic/train.csv')

test= pd.read_csv('../input/titanic/test.csv')
train.info()
x_test = train['Survived']

x_train = train.drop(['PassengerId','Survived','Cabin','Name'],axis=1)

y_train = test.drop(['PassengerId','Cabin','Name'],axis=1)
y_train.info() 
#将x_train和y_train合并以便统一进行编码

feature = pd.concat([x_train,y_train],ignore_index=True)
#利用标签编码方式将object对象转化为数值类型

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

fea = feature.dropna()

for i in range(fea.shape[1]):

    if fea.iloc[:,i].dtype == 'object':

        fea.iloc[:,i] = pd.Series(label_encoder.fit_transform(fea.iloc[:,i].astype(str)))

fea.corr(method='pearson')
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

fig,axes = plt.subplots(1,9,figsize=(20,3))

sns.distplot(fea['Age'],ax=axes[0])

[axes[i+1].scatter(fea['Age'],fea.iloc[:,i]) for i in range(8)]
mean = feature['Age'].dropna().mean()

std = feature['Age'].dropna().std()

fill_nan = pd.Series(np.abs(np.random.normal(mean,std,feature['Age'].isnull().sum())),index=feature[feature['Age'].isnull()==True].index)
for i in fill_nan.index:

    feature['Age'][i] = fill_nan[i] 

feature.info()
fig,axes = plt.subplots(1,9,figsize=(20,3))

sns.distplot(fea['Fare'],ax=axes[0])

[axes[i+1].scatter(fea['Fare'],fea.iloc[:,i]) for i in range(8)]
feature['Fare'] = feature['Fare'].fillna(feature['Fare'].median())
sns.countplot(feature['Embarked'].dropna())
#对Embarked缺失值也选用众数填充

feature['Embarked'] = feature['Embarked'].fillna('S')
#为消除大量缺失值带来的误差，这里将age进行数据规约

feature['age_level'] = pd.cut(feature['Age'],bins=[0,6,12,17,45,69,100],labels=['0-6','7-12','13-17','18-45','46-69','70-100'])
feature.info()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



for i in range(feature.shape[1]):

    if feature.iloc[:,i].dtype == 'object'or'category':

        feature.iloc[:,i] = pd.Series(label_encoder.fit_transform(feature.iloc[:,i].astype(str)))
X_train = feature.iloc[:x_train.shape[0],:].drop('Age',axis=1)

y_train = feature.iloc[x_train.shape[0]:,:].drop('Age',axis=1)
y_train.head()
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=90,max_depth=5,max_features=1,class_weight='balanced',random_state=17)

forest.fit(X_train,x_test)
forest.score(X_train,x_test)
Survived = forest.predict(y_train)

df = pd.DataFrame({'PassengerId':y_train.index.values+1,'Survived':Survived})

df.to_csv('submission.csv',index=False)

df.head()