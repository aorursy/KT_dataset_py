import pandas as pd

from pandas import Series,DataFrame



import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from matplotlib import font_manager, rc



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv", index_col='PassengerId')

test= pd.read_csv("../input/test.csv", index_col='PassengerId')



# preview the data



print(train.shape)

train.head()
combi = pd.concat([train, test])

print(combi.shape)

combi.head()
# drop unnecessary columns, these columns won't be useful in analysis and prediction

combi = combi.drop(['Name','Ticket'], axis=1)

print(combi.shape)

combi.head()
combi.info()



# info() 를 각 column의 데이터 타입과 NaN 포함 여부를 확인.

# Embarked column은 1309- 1307 = 2개의 NaN value를 포함
combi[pd.isnull(combi['Embarked'])]
 # confirmation top freqeunce value

combi['Embarked'].describe()
 # fill the 'NaN' value of 'Embarked' column

combi['Embarked'] = combi['Embarked'].fillna('S')

combi[pd.isnull(combi['Embarked'])]
sns.factorplot('Embarked','Survived', hue="Sex", data=combi, size=4, aspect=3)



# (x data, y data, total dataframe, 크기, 크기)

# barplot이나 scatter 과 다르게 "비교"를 위해 만들어진 plot이다. 

# kind 인자를 통해 plot 모양의 선택이 가능하다.

# factorplot의 가장 큰 장점은 'categorical' data를 손쉽게 비교하고 분석 할 수 있다는 것이다.

# 그리고 FacetGrid가 매우 간편한데. col argument에 'Day'를 입력했다하면 날짜별 plot을 나눠 볼 수 있다.
fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked', data=combi, ax=axis1)

sns.countplot(x='Survived', hue='Embarked', data=combi, order=[1,0], ax=axis2)

# countplot은 data의 개수를 counting한다.



embark_perc = combi[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()

#  Embarked column을 기준으로 group을 묶는다. 



sns.barplot(x='Embarked',y='Survived', data=embark_perc, order=['S','C','Q'], ax= axis3)
embarked_dummies = pd.get_dummies(combi['Embarked'], prefix='Embarked')

embarked_dummies.drop(['Embarked_S'], axis=1, inplace=True)
combi = pd.concat([combi, embarked_dummies], axis=1)

combi.drop('Embarked',axis=1,inplace=True)

print(combi.shape)

combi.head()
print(np.dtype(combi['Fare']))

combi[pd.isnull(combi['Fare'])]
combi['Fare'].fillna(combi['Fare'].median(), inplace=True)

# NaN 값처리



combi['Fare']=combi['Fare'].astype(np.int)

# float 자료형을 int로 변경



fare_not_survived = combi['Fare'][combi['Survived']==0]

fare_survived = combi['Fare'][combi['Survived']==1]

# 'Fare' 별 생존자 수 비교



average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

# DataFrame을 사용하여 list타입을 변경해 주었다. ( visualization을 위해 )



combi['Fare'].plot(kind='hist',  bins=100, xlim=(0,50), figsize=(15,3))

# 시각화

#!! histogram ( x축의 범위가 넓다 )을 그릴때는 sns 가 아닌 plot kind='hist' 가 좋다. or data.hist()



average_fare.index.names = std_fare.index.names = ['Survived']

average_fare.plot(yerr=std_fare, kind='bar', legend=False, title='Survival rate of Fare', ecolor='red')

# plot 함수의 사용     <->  plt.bar 의 사용

# yerr 는 해당 plot 마다 'y축'으로 선언해준 값만큼 선을 그려준다 => 평균과 분산을 볼때 자주 사용된다.
fig , (axis1, axis2) = plt.subplots(1,2, figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



average_age = combi['Age'].mean()

std_age = combi['Age'].std()

count_nan_age = combi['Age'].isnull().sum()

# null값의 개수를 counting 하는 skill



rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# random 변수 생성

# x 부터 y-1 까지 포함 size는 개수





combi['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# NaN 값들을 drop시키고 남은 값들을 int시킨다



combi['Age'][pd.isnull(combi['Age'])] = rand_1

# NaN 값들의 빈값을 평균과 분산을 통해 처리하였다.



combi['Age'] = combi['Age'].astype(int)



combi['Age'].hist(bins=70, ax=axis2)
facet = sns.FacetGrid(combi, hue='Survived', aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, combi['Age'].max()))

facet.add_legend()

facet.set_titles('Line about surviving by their age')



fig, (axis1,axis2) = plt.subplots(2,1, figsize=(18,8))

average_age = combi[['Age','Survived']].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age, ax=axis1)

# average_age 변수에 2개의 column( age, survived ) 가 있다.



sns.countplot(x='Age', data=combi, ax=axis2)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age<=13 else sex



combi['Person'] = combi[['Age', 'Sex']].apply(get_person, axis=1)



combi.drop(['Sex'], axis=1).shape

#drop already existing column



fig, (axis1, axis2) = plt.subplots(1,2, figsize=(15,4))



sns.countplot(x='Person', data=combi, ax=axis1)

# count visual



person_prec = combi[['Person','Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_prec, ax=axis2, order=['male','female','child'])

# percentage visual

person_dummies = pd.get_dummies(combi['Person'])

person_dummies.columns = ['Child','Female','Male']

person_dummies.drop(['Male'], axis=1, inplace=True)

# Male 변수를 Family에서 제거
combi = pd.concat([combi, person_dummies], axis=1)

print(combi.shape)

combi.head()
combi.drop(['Sex','Person'], axis=1, inplace=True)

print(combi.shape)

combi.head()
combi[pd.isnull(combi['Cabin'])].shape
combi.drop('Cabin',axis=1,inplace=True)
combi['Family'] = combi['Parch'] + combi['SibSp']

# combinate two columns
combi['Family'].loc[combi['Family'] > 0] = 1

combi['Family'].loc[combi['Family'] == 0] = 0

# Family feature composed of 0 or 1

# Family value 1 mean that this row have a family member 

# Family value 0 mean that this row have not a family member  
combi.drop(['Parch','SibSp'], axis=1, inplace=True)



fig, (axis1,axis2) = plt.subplots(1,2, sharex=True, figsize=(10,5) )



sns.countplot(x='Family', data=combi, order=[1,0], ax=axis1)



family_perc = combi[['Family','Survived']].groupby(['Family'], as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

#시각화



axis1.set_xticklabels(['With Family','Alone'], rotation=0)
sns.factorplot('Pclass','Survived',order=[1,2,3], data=combi, size=5)

# 3의 생존률이 매우 떨어진다.



pclass_dummies = pd.get_dummies(combi['Pclass'], prefix='Class')

pclass_dummies.drop('Class_3', axis=1, inplace=True)



combi.drop(['Pclass'], axis=1, inplace=True)
combi = pd.concat([combi, pclass_dummies], axis=1)

print(combi.shape)

combi.head()
train = combi[pd.notnull(combi['Survived'])]

print(train.shape)



test = combi[pd.isnull(combi['Survived'])]

print(test.shape)
# train data featrue selection



feature_names = ['Embarked_C' ,'Embarked_Q', 'Child' ,'Female' ,'Family', 'Class_1' ,'Class_2']



X_train = train[feature_names]



print(X_train.shape)

X_train.head()
# train data "tareget" feature selection



label_name = 'Survived'

y_train = train[label_name]



print(y_train.shape)

y_train.head()
from sklearn.cross_validation import cross_val_score



model = RandomForestClassifier(n_estimators=100)

score = cross_val_score(model, X_train, y_train, cv=100).mean()



print('Score = {score: .5f}' .format(score=score))
X_test = test[feature_names]

model.fit(X_train, y_train)



prediction = model.predict(X_test)

prediction[:15]
submission = pd.read_csv('../input/gendermodel.csv', index_col='PassengerId')



submission['Survived'] = prediction.astype(np.int)



print(submission.shape)

submission.head()