# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd #Data analysis package
import numpy as np #Scientific computing package
from pandas import Series,DataFrame
data_train = pd.read_csv("../input/train.csv")
data_train  #Output view
data_train.info() #Output view,Check out general information
data_train.describe() #Check out general information
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.6)  
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Rescue (1 as survival)") 
plt.ylabel(u"Number")
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"Number")
plt.title(u"Pclass")
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"Age")                         
plt.grid(b=True, which='major', axis='y') 
plt.title(u"rescue distribution depends on Age(1 as survival)")
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")
plt.ylabel(u"density") 
plt.title(u"Age distribution of passengers at all ranks")
plt.legend((u'Pclass 1', u'Pclass 2',u'Pclass 3'),loc='best') 
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Number of Embarked ")
plt.ylabel(u"Number")
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'survival':Survived_1, u'death':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"survival of passengers rank")
plt.xlabel(u"Passenger rank") 
plt.ylabel(u"Number")
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"survival of sex")
plt.xlabel(u"Sex") 
plt.ylabel(u"Number")
fig=plt.figure()
fig.set(alpha=0.6) 
plt.title(u"Survival according to Pclass and sex")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"Survival", u"Death"], rotation=0)
ax1.legend([u"Female /highclass"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"Death", u"Survival"], rotation=0)
plt.legend([u"Female/low class"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"Death", u"Survival"], rotation=0)
plt.legend([u"Male/High Class"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"Death", u"Survival"], rotation=0)
plt.legend([u"Male/Low Class"], loc='best')
plt.show()
fig = plt.figure()
fig.set(alpha=0.6)  
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survival':Survived_1, u'Death':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Passengers rescued at each port of entry")
plt.xlabel(u"Embarked ") 
plt.ylabel(u"Number") 
plt.show()
rslt = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(rslt.count()['PassengerId'])
df
data_train.Cabin.value_counts()
fig = plt.figure()
fig.set(alpha=0.6)
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'yes':Survived_cabin, u'no':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"Survived or not")
plt.xlabel(u"Cabin or not") 
plt.ylabel(u"Number")
plt.show()
from sklearn.ensemble import RandomForestRegressor
# use  RandomForestClassifier to fill in the missing age value
def set_missing_ages(df):
    # put them to Random Forest Regressor
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    #  known age and unknown age
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y as Age
    y = known_age[:, 0]
    # X as Characteristic attribute value
    X = known_age[:, 1:]
    # fit to RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    #  predict the age of unknown age
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # Fill in the missing data with the predicted results.
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df, rfr
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df
import sklearn.preprocessing as preprocessing
import numpy as np
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
df
from sklearn import linear_model
# take out the value
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
# y as Survival
y = train_np[:, 0]
# X as Characteristic attribute value
X = train_np[:, 1:]
# fit to RandomForestRegressor
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
clf
data_test = pd.read_csv("../input/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# the same as above,andomForestRegressor
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
#filling predictedAges according to Characteristic attribute
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
df_test
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
#result.to_csv("../input/Logistic_Regression_Predictions.csv", index=False)
print("Done")