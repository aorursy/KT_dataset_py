import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")

import time



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split, ShuffleSplit,cross_validate, StratifiedKFold



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC



from xgboost import XGBClassifier



from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam, SGD, Adamax

from keras.layers.normalization import BatchNormalization



import h5py
#讀取文件

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.info()

#檢查資料中的遺失值數目，以此來考慮如何處理
test_df.info()

#檢查資料中的遺失值數目，以此來考慮如何處理
test_id = test_df['PassengerId']

#保存test樣本的乘客id
test_df.isnull().sum()
train_df.isnull().sum()
dataset = pd.concat(objs=[train_df,test_df],axis=0).reset_index(drop=True)

#將training testing data串在一起，方便做資料清洗
dataset.fillna(np.nan)
survived = dataset[dataset['Survived']==1]

nonsur = dataset[dataset['Survived'] == 0]

plt.figure(figsize=(12,10))

plt.subplot(331)

sns.distplot(survived['Age'].dropna().values,

             bins=range(0,81,1), kde=False, color='b')

sns.distplot(nonsur['Age'].dropna().values,

              bins=range(0,81,1), kde =False, color='r')

plt.subplot(332)

sns.barplot('Sex', 'Survived', data=dataset)

plt.subplot(333)

sns.barplot('Pclass', 'Survived', data=dataset)

plt.subplot(334)

sns.barplot('Embarked', 'Survived', data=dataset)

plt.subplot(335)

sns.barplot('SibSp', 'Survived', data=dataset)

plt.subplot(336)

sns.barplot('Parch', 'Survived', data=dataset)

plt.subplot(337)

sns.distplot(np.log(survived['Fare'].dropna().values+1), kde=False, color='b',axlabel='Fare')

sns.distplot(np.log(nonsur['Fare'].dropna().values+1), kde=False, color='r')

 
#針對age變量進行分析

g = sns.FacetGrid(train_df,col='Survived')

g = g.map(sns.distplot,'Age')
g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 0) & (train_df['Survived'].notnull())] , color='blue', shade=True)

g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 1) & (train_df['Survived'].notnull())], color='red', shade=True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
g = sns.factorplot(y='Age',x='Sex', data=dataset,kind='box')

g = sns.factorplot(y='Age',x='Sex',hue='Pclass',data=dataset,kind='box')

#觀察age變量是否會受其他變量的影響，結果證明有此影響(Pclass)
#創建新特徵--頭銜

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset.groupby('Title')['Age'].describe()
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Mrs":2, "Mr":3, "Rare":4})

dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==0), 'Age']=6

dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==1), 'Age']=22

dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==2), 'Age']=37

dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==3), 'Age']=33

dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==4), 'Age']=46

#為頭銜變量作編號，並根據頭銜來填補其age的遺失值
#檢查Age的遺失值是否填補完畢

dataset.isnull().sum()
#檢查Embarkde遺失值的情況

print(dataset[dataset['Embarked'].isnull()])
#針對Embarked變量作分析

plt.figure(figsize=(20,15))

plt.subplot(221)

sns.countplot('Embarked', data=dataset)

plt.subplot(222)

sns.countplot('Embarked', hue='Pclass', data=dataset)

plt.subplot(223)

sns.countplot('Embarked', hue='Sex',data=dataset)

plt.subplot(224)

sns.countplot('Embarked', hue='Survived',data=dataset)
#針對Embarked填補遺失值

dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(dataset[dataset['Fare'].isnull()])
#針對Fare填補遺失值

dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'][dataset['Pclass']==3].mean())
dataset.info()
#創建新特徵 -- Age_cat

#將年齡從continous values 改成 categorical values

dataset['Age_cat']=0

dataset.loc[dataset['Age']<=15,'Age_cat']=0

dataset.loc[(dataset['Age']>15)&(dataset['Age']<=30),'Age_cat']=1

dataset.loc[(dataset['Age']>30)&(dataset['Age']<=45),'Age_cat']=2

dataset.loc[(dataset['Age']>45)&(dataset['Age']<=60),'Age_cat']=3

dataset.loc[dataset['Age']>60,'Age_cat']=4
sns.factorplot(x='Age_cat', y='Survived', data=dataset)
#創建新特徵---家族數目

dataset["Familysize"] = dataset["SibSp"] + dataset["Parch"] 
g = sns.factorplot(x="Familysize",y="Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
dataset['Familysize_cat'] = pd.cut(dataset['Familysize'], 4)

dataset.groupby(dataset['Familysize_cat'])['Survived'].describe()
#將家族數目的值改為類別

dataset['Familysize_cat'] = 0

dataset.loc[(dataset['Familysize'] > 0.1) & (dataset['Familysize'] <= 2.5), 'Familysize_cat']  = 0

dataset.loc[(dataset['Familysize'] > 2.5) & (dataset['Familysize'] <= 5.0), 'Familysize_cat'] = 1

dataset.loc[(dataset['Familysize'] > 5.0) & (dataset['Familysize'] <= 7.5), 'Familysize_cat']   = 2

dataset.loc[ dataset['Familysize'] > 7.5, 'Familysize_cat']  = 3

dataset['Familysize_cat'] = dataset['Familysize_cat'].astype(int)
g = sns.factorplot(x='Familysize_cat',y='Survived',data=dataset,kind='bar')

g.set_ylabels('Survival Probability')
dataset['IsAlone'] = np.where(dataset['Familysize'] ==0 ,1,0)
sns.factorplot(x='IsAlone', y='Survived', data=dataset)
dataset['Fare_R'] = pd.qcut(dataset['Fare'],4)

dataset.groupby(dataset['Fare_R'])['Survived'].describe()
dataset['Fare_cat'] = 0

dataset.loc[(dataset['Fare'] <= 7.896), 'Fare_cat']  = 0

dataset.loc[(dataset['Fare'] > 7.896) & (dataset['Fare'] <= 14.454), 'Fare_cat'] = 1

dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.275), 'Fare_cat']   = 2

dataset.loc[ dataset['Fare'] > 31.275, 'Fare_cat']  = 3

dataset['Fare_cat'] = dataset['Fare_cat'].astype(int)
sns.factorplot(x='Fare_cat',y='Survived',data=dataset)
dataset['Shared_ticket'] = np.where(dataset.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)

sns.barplot('Shared_ticket', 'Survived', data=dataset)

#可以發現團體出遊的話，存活率的確比較高

print(dataset.groupby('Ticket'))
dataset['Cabin_known'] = dataset['Cabin'].isnull()==False

g = sns.barplot('Cabin_known', 'Survived', data=dataset)
g = sns.factorplot('Cabin_known', 'Survived',hue='Sex',data=dataset)

g = sns.factorplot('Cabin_known', 'Survived',hue='Pclass',data=dataset)
dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
dataset = pd.get_dummies(dataset, columns = ['Title'])

dataset = pd.get_dummies(dataset, columns = ['Embarked'],prefix='Em')
dataset.info()
#拋棄不需要的特徵

drop_element =['Fare_R','Age','Fare','Cabin', 'PassengerId', 'Familysize', 'Parch', 'SibSp','Ticket','Name']

dataset = dataset.drop(drop_element, axis=1)
dataset.info()
dataset.sample()
#分解數據集

train_len = len(train_df)

train = dataset[:train_len]

test = dataset[train_len:]

test.drop('Survived', axis=1, inplace=True)
train.info()
test.info()
#將數據集轉換成訓練集

train["Survived"] = train["Survived"].astype(int)

train_y = train["Survived"]

train_y = train_y.values

train_x = train.drop("Survived",axis = 1)

train_x = train_x.values
#利用keras構築類神經網路、用kfold做cross validation



filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

callbacks_list = [checkpoint]



adamax01 = Adamax(lr=0.0002)

def model1():

    model = Sequential()

    model.add(Dense(units=256, input_dim=16, kernel_initializer='lecun_normal', activation='selu'))  

    model.add(BatchNormalization())

    model.add(Dense(units=128, kernel_initializer='lecun_normal', activation='selu'))  

    model.add(BatchNormalization())

    model.add(Dense(units=1, kernel_initializer='lecun_normal', activation='sigmoid'))  

    model.compile(loss='binary_crossentropy',

                  optimizer=adamax01,

                  metrics=['accuracy'])

    return model



model1 = model1()



cvscore = []

K=10

folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=0).split(train_x, train_y))

for j, (train_idx, test_idx) in enumerate(folds):

    print('\n===================FOLD=',j)

    x_train_cv = train_x[train_idx]

    y_train_cv = train_y[train_idx]

    x_hold = train_x[test_idx]

    y_hold = train_y[test_idx]



    model1.fit(x_train_cv, y_train_cv, epochs=300, validation_data=([x_hold, y_hold]),

              batch_size=32, verbose=0, callbacks=callbacks_list)

    score = model1.evaluate(x_hold,y_hold)

    cvscore.append(score[1])

    print('acc:',score[1])
print('mean:',np.mean(cvscore), 'std:',np.std(cvscore))
plt.plot(cvscore)

plt.ylabel('acc')

plt.xlabel('round')
model1.load_weights(filepath=filepath)
#製作提交文檔

predictions = model1.predict(test)

predictions = np.round(predictions)

predictions = predictions.flatten()

Submission = pd.DataFrame({ 'PassengerId': test_id ,

                            'Survived': predictions.astype(int)})

Submission.to_csv("Submission.csv", index=False)



Submission.head()