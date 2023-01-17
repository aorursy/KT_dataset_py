import pandas as pd

import sklearn as sk

import numpy as np



%pylab inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.describe()
data_train.shape
data_train.Survived.value_counts()
from IPython.core.display import HTML

HTML('<title>EDA</title><br><h1>Learn about the data!</h1><h2>For each variable:</h2><br><li>Is the variable categorical?<br><li>If not, Min Max and Average values?<br><li>If yes,what are the categories?<br><li>are there missing values?<br><li>Info about the distribution of the variable')
data_train.Sex.value_counts()
data_train.Sex.value_counts().plot(kind='bar')
data_train[data_train.Sex == 'female']
data_train[data_train.Sex.isnull()]
data_train.Fare.value_counts()
data_train.Fare.hist(bins=5)
data_test[data_test.Fare.isnull()]

data_test['Fare'] = data_test['Fare'].fillna(value=data_test.Fare.mean())

data_test[data_test.Fare.isnull()]
data_train[data_train.Fare == 0]
data_test[data_test.Cabin.isnull()]
#women and children first?

fig, axs = plt.subplots(1,2)

data_train[data_train.Sex == 'male'].Survived.value_counts().plot(kind='barh',ax=axs[0],title="Male Survivorship")

data_train[data_train.Sex == 'female'].Survived.value_counts().plot(kind='barh',ax=axs[1],title="Male Survivorship")
data_train[data_train.Age<15].Survived.value_counts().plot(kind='barh')
data_train[(data_train.Age<15)&(data_train.Sex =='female')].Survived.value_counts().plot(kind='barh')
data_train[(data_train.Age<15)&(data_train.Sex =='male')].Survived.value_counts().plot(kind='barh')
data_test[data_test.Age.isnull()]
avg_age = data_train.Age.mean()

avg_tes = data_test.Age.mean()
data_train.Age = data_train.fillna(value=avg_age)

data_test.Age = data_test.fillna(value=avg_tes)
data_train[data_train.Age.isnull()]

data_test[data_test.Age.isnull()]

#data_train['Age'].fillna(X.Age.mean(),inplace=True)
data_train['Family_Size'] = data_train['SibSp']+data_train['Parch']

data_test['Family_Size'] = data_test['SibSp']+data_test['Parch']

fig,axs = subplots(1,2)

data_train[data_train.Survived == 1].Family_Size.value_counts().plot(kind='bar',ax=axs[0],title='Family survived')

data_train[data_train.Survived == 0].Family_Size.value_counts().plot(kind='bar',title='Family not survived')
data_train['Family_Size'].value_counts()

data_test['Family_Size'].value_counts()
"""

######################################################

def function(data_train):

    for row in data_train.iterrows():

        if row['Family_Size'] == 0:

            val = 1

        else:

            val = 0

    return val



data_train['Is_Alone'] = data_train.apply(function, axis=1)

#######################################################



"""

def function(row):

    val = 0

    if row['Family_Size'] == 0:

        val = 1

    else:

        val = 0

    return val



data_train['Is_Alone'] = data_train.apply(lambda row : function(row),axis=1)

data_test['Is_Alone'] = data_test.apply(lambda row : function(row),axis=1)
data_train['Is_Alone'].value_counts()

data_test['Is_Alone'].value_counts()
fig, axs = subplots(1,2)

data_train[data_train.Is_Alone == 1].Survived.value_counts().plot(kind='bar',ax = axs[0],title='Alone Passenger')

data_train[data_train.Is_Alone == 0].Survived.value_counts().plot(kind='bar',ax = axs[1],title='Not alone Passenger')
data_train.pop('SibSp')

data_train.pop('Parch')

data_test.pop('SibSp')

data_test.pop('Parch')

data_train.describe()

data_test.describe()
fig, axs = subplots(1,2)

data_train[data_train.Survived == 1].Fare.value_counts().plot(kind='bar',ax = axs[0],title='Survived Passenger')

data_train[data_train.Survived == 0].Fare.value_counts().plot(kind='bar',ax = axs[1],title='Not Survived Passenger')
fix, axs = subplots(1,3)

data_train[data_train.Pclass == 1].Survived.value_counts().plot(kind='barh',ax=axs[0],title='PClass 1 survival rate')

data_train[data_train.Pclass == 2].Survived.value_counts().plot(kind='barh',ax=axs[1],title='PClass 2 survival rate')

data_train[data_train.Pclass == 3].Survived.value_counts().plot(kind='barh',ax=axs[2],title='PClass 3 survival rate')
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer(neg_label=0,pos_label=1,sparse_output=False)

lb.fit_transform(data_train['Sex'])

print (lb.classes_)



data_train.Sex
data_train['Sex'][data_train['Sex'] == 'male'] = 0

data_train['Sex'][data_train['Sex'] == 'female'] = 1



data_test['Sex'][data_test['Sex'] == 'male'] = 0

data_test['Sex'][data_test['Sex'] == 'female'] = 1



data_train['Embarked'][data_train['Embarked'] == 'S'] = 0

data_train['Embarked'][data_train['Embarked'] == 'C'] = 1

data_train['Embarked'][data_train['Embarked'] == 'Q'] = 2



data_test['Embarked'][data_test['Embarked'] == 'S'] = 0

data_test['Embarked'][data_test['Embarked'] == 'C'] = 1

data_test['Embarked'][data_test['Embarked'] == 'Q'] = 2



data_train[data_train.Embarked.isnull()]

data_train.dtypes
data_train[data_train.Sex.isnull()]

data_train.Embarked = data_train.fillna(value=data_train.Embarked.mean())

data_train['Age'].astype(str).astype(int)

data_train['Sex'].astype(str).astype(int)

data_train['Embarked'].astype(str).astype(int)

data_feature =data_train[['Age','Pclass','Is_Alone','Family_Size','Embarked','Sex','Survived']]

data_feature['Embarked'] = pd.to_numeric(data_feature['Embarked'], errors='ignore')

data_feature['Sex'] = pd.to_numeric(data_feature['Sex'], errors='ignore')



data_feature.dtypes
data_test[data_test.Sex.isnull()]

data_test.Embarked = data_test.fillna(value=data_test.Embarked.mean())

data_test['Age'].astype(str).astype(int)

data_test['Sex'].astype(str).astype(int)

data_test['Embarked'].astype(str).astype(int)

test_feature =data_test[['Age','Pclass','Is_Alone','Family_Size','Embarked','Sex']]

test_feature['Embarked'] = pd.to_numeric(test_feature['Embarked'], errors='ignore')

test_feature['Sex'] = pd.to_numeric(test_feature['Sex'], errors='ignore')



test_feature.dtypes
x_train, x_validate, x_test = np.split(data_feature.sample(frac=1), [int(.6*len(data_feature)), int(.8*len(data_feature))])

y_train = x_train.pop('Survived')

y_validate = x_validate.pop('Survived')

y_test = x_test.pop('Survived')



from sklearn.preprocessing import normalize

x_train = normalize(x_train)

x_validate = normalize(x_validate)

x_test = normalize(x_test)

test_feature = normalize(test_feature)
from sklearn.model_selection import learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier



clf_knn = KNeighborsClassifier()

clf_log = LogisticRegression()

clf_svc = SVC()

clf_rbf = SVC(kernel='rbf')

clf_ran = RandomForestClassifier()





clf_knn.fit(x_train,y_train)

y_val = clf_knn.predict(x_validate)

score = accuracy_score(y_validate,y_val)

print (score)

clf_knn.fit(x_validate,y_validate)

y_label = clf_knn.predict(x_test)

score = accuracy_score(y_test,y_label)

print (score)

clf_knn.fit(x_test,y_test)

test_label = clf_knn.predict(test_feature)
clf_log.fit(x_train,y_train)

y_val = clf_log.predict(x_validate)

score = accuracy_score(y_validate,y_val)

print (score)

clf_log.fit(x_validate,y_validate)

y_label = clf_log.predict(x_test)

score = accuracy_score(y_test,y_label)

print (score)

clf_log.fit(x_test,y_test)

test_label = clf_log.predict(test_feature)
clf_svc.fit(x_train,y_train)

y_val = clf_svc.predict(x_validate)

score = accuracy_score(y_validate,y_val)

print (score)

clf_svc.fit(x_validate,y_validate)

y_label = clf_svc.predict(x_test)

score = accuracy_score(y_test,y_label)

print (score)

clf_svc.fit(x_test,y_test)

test_label = clf_svc.predict(test_feature)
clf_rbf.fit(x_train,y_train)

y_val = clf_rbf.predict(x_validate)

score = accuracy_score(y_validate,y_val)

print (score)

clf_rbf.fit(x_validate,y_validate)

y_label = clf_rbf.predict(x_test)

score = accuracy_score(y_test,y_label)

print (score)

clf_rbf.fit(x_test,y_test)

test_label = clf_rbf.predict(test_feature)
clf_ran.fit(x_train,y_train)

y_val = clf_ran.predict(x_validate)

score = accuracy_score(y_validate,y_val)

print (score)

clf_ran.fit(x_validate,y_validate)

y_label = clf_ran.predict(x_test)

score = accuracy_score(y_test,y_label)

print (score)

clf_ran.fit(x_test,y_test)

test_label = clf_ran.predict(test_feature)
data_test.assign(Survived = test_label)

data_test.to_csv("submission.csv", index=False)

print ("Done")