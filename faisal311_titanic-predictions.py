import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection,preprocessing,linear_model,metrics

import numpy as np

import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

sub_data = pd.read_csv('../input/titanic/gender_submission.csv')
train_data.describe()
train_data.Age.fillna(value = train_data['Age'].mean(),inplace = True)
def clean_data(data):

    data.loc[data['Sex']=='male','Sex'] = 0

    data.loc[data['Sex']=='female','Sex'] = 1

    data.loc[data['Embarked']=='S','Embarked'] = 0

    data.loc[data['Embarked']=='C','Embarked'] = 1

    data.loc[data['Embarked']=='Q','Embarked'] = 2

    



clean_data(train_data)
train_data.Survived.value_counts(normalize = True).plot(kind='bar',alpha = 0.6)

plt.title('Survived')



plt.show()



female_color = 'coral'

male_color = 'grey'
plt.scatter(train_data.Survived,train_data.Age, alpha = 0.1)

plt.title('Age wrt Survived')

plt.show()



train_data.Pclass.value_counts().plot(kind='bar')
for x in [1,2,3]:

    train_data.Age[train_data.Pclass == x].plot(kind='kde')

plt.title('Class wrt Age')

plt.legend(('1st','2nd','3rd'))

plt.show()
train_data.Survived[train_data.Sex==0].value_counts().plot(kind = 'bar',color = male_color)

plt.title('Male survived')

plt.show()
train_data.Survived[train_data.Sex==1].value_counts().plot(kind = 'bar',color = female_color)

plt.title('Female survived')

plt.show()
train_data.Sex[train_data.Survived==1].value_counts().plot(kind='bar',color=(female_color,male_color),alpha = 0.5)

plt.title('Survived wrt Sex')

plt.show()
for x in [1,2,3]:

    train_data.Survived[train_data.Pclass==x].plot(kind='kde')

plt.title('Survived wrt Class')

plt.legend(('1st','2nd','3rd'))

plt.show()
train_data.Survived[(train_data.Pclass == 1) & (train_data.Sex == 0)].value_counts(normalize = True).plot(kind='bar',color = male_color)

plt.title('Rich Men Survived')

plt.show()
train_data.Survived[(train_data.Pclass == 3) & (train_data.Sex == 0)].value_counts(normalize = True).plot(kind='bar',color = male_color)

plt.title('Poor Men Survived')

plt.show()
train_data.Survived[(train_data.Pclass == 1) & (train_data.Sex == 1)].value_counts(normalize = True).plot(kind='bar',color= female_color,alpha = 0.5)

plt.title('Rich women Survived')

plt.show()
train_data.Survived[(train_data.Pclass == 3) & (train_data.Sex == 1)].value_counts(normalize = True).plot(kind='bar',color = female_color,alpha = 0.5)

plt.title('Poor women Survived')

plt.show()
# Most people embarked at S

train_data['Embarked'].value_counts(normalize = True)
# We can assume the NaN values to be S

train_data.Embarked.fillna(value = 0,inplace=True)
train_data['Age range'] = pd.cut(train_data['Age'],5)

train_data['Age range'].value_counts()
def clean_age(data):

    data.loc[data['Age'] <= 16,'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32),'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48),'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64),'Age'] = 3

    data.loc[(data['Age'] > 64) & (data['Age'] <= 80),'Age'] = 4

clean_age(train_data)
# Shows the survival rate of younger people is higher

train_data[['Age range','Survived']].groupby('Age range').mean()
train_data['Fare range'] = pd.cut(train_data['Fare'],5)
train_data['Fare range'].value_counts()
def clean_Fare(data):

    data.loc[data['Fare'] <= 102,'Fare'] = 0

    data.loc[(data['Fare'] > 102) & (data['Fare'] <= 204),'Fare'] = 1

    data.loc[(data['Fare'] > 204) & (data['Fare'] <= 307),'Fare'] = 2

    data.loc[(data['Fare'] > 307) & (data['Fare'] <= 409),'Fare'] = 3

    data.loc[(data['Fare'] > 409),'Fare'] = 4

clean_Fare(train_data)
train_data[['SibSp','Survived']].groupby('SibSp').mean()
train_data[['Parch','Survived']].groupby('Parch').mean()
# Class 1 had higher survival rate than 2 & 3

train_data[['Pclass','Survived']].groupby('Pclass').mean()
# Final check

train_data.head()
test_data.head()
test_data.describe()
test_data['Age'].fillna(value=test_data['Age'].mean(),inplace=True)

test_data['Fare'].fillna(value = 0,inplace=True)
clean_data(test_data)

clean_age(test_data)

clean_Fare(test_data)
target = 'Survived'



features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
regr = linear_model.LogisticRegression()

regr_model = regr.fit(train_data[features],train_data[target])

print(regr_model.score(train_data[features],train_data[target]))
regr_pred = regr_model.predict(test_data[features])
print(metrics.mean_absolute_error(sub_data['Survived'],regr_pred))



print(metrics.mean_squared_error(sub_data['Survived'],regr_pred))



print(np.sqrt(metrics.mean_squared_error(sub_data['Survived'],regr_pred)))
poly = preprocessing.PolynomialFeatures(degree=2)

poly_features = poly.fit_transform(train_data[features])



regr_model = regr.fit(poly_features,train_data[target])

print(regr_model.score(poly_features,train_data[target]))
test_poly_features = poly.fit_transform(test_data[features])

poly_regr_pred = regr_model.predict(test_poly_features)
print(metrics.mean_absolute_error(sub_data['Survived'],poly_regr_pred))



print(metrics.mean_squared_error(sub_data['Survived'],poly_regr_pred))



print(np.sqrt(metrics.mean_squared_error(sub_data['Survived'],poly_regr_pred)))
output_regr = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':regr_pred})
output_regr.to_csv('Titanic Prediction.csv',index=0)