import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16



data = pd.read_csv('../input/train.csv')
data.head()
plt.hist(data['Survived'])

plt.title('Survivors')

plt.ylabel('Total number of people')

plt.xticks([0,1],('Dead','Alive'))

plt.xlabel('')

plt.show()
plt.hist(data[data['Sex']=='male']['Survived'])

plt.title('Male survivors')

plt.ylabel('Number of men')

plt.xticks([0,1],('Dead','Alive'))

plt.xlabel('')

plt.show()
plt.hist(data[data['Sex']=='female']['Survived'])

plt.title('Female survivors')

plt.ylabel('Number of women')

plt.xticks([0,1],('Dead','Alive'))

plt.xlabel('')

plt.show()
data.hist()

plt.show()
correlations = data.corr()

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

largeval = max(correlations.values.flatten()[correlations.values.flatten()<1])

smallval = min(correlations.values.flatten())

cax = ax.matshow(correlations, vmin=smallval, vmax=largeval)

fig.colorbar(cax)

ticks = np.arange(0,7,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(list(correlations),rotation=90)

ax.set_yticklabels(list(correlations))

plt.show()
correlations
data.head()
redata = data.copy()

redata['HasCabin'] = 1.0*np.invert(data['Cabin'].isnull())

del redata['Cabin']

del redata['Ticket']

del redata['Name']

redata['HasAgeRecorded'] = 1.0*np.invert(redata.Age.isnull())

redata['Age'] = redata['Age'].fillna(0)

redata['Embarked'].replace(to_replace='S',value=1.,inplace=True)

redata['Embarked'].replace(to_replace='C',value=2.,inplace=True)

redata['Embarked'].replace(to_replace='Q',value=3.,inplace=True)

redata['Embarked'].fillna(0,inplace=True)

redata['Sex'].replace(to_replace='male',value=0.,inplace=True)

redata['Sex'].replace(to_replace='female',value=1.,inplace=True)

redata.head()
from sklearn.preprocessing import StandardScaler

scale_fare = StandardScaler()

scale_age = StandardScaler()

scale_sib = StandardScaler()

scale_parch = StandardScaler()

scale_sib = StandardScaler()

scale_emb = StandardScaler()

scale_class = StandardScaler()

scale_sex = StandardScaler()

redata['Fare'] = scale_fare.fit_transform(redata.Fare.values.reshape(-1,1))

redata['Age'] = redata['Age']/np.max(redata['Age'].values.reshape(-1,1))

redata['SibSp'] = scale_sib.fit_transform(redata.SibSp.values.reshape(-1,1))

redata['Parch'] = scale_parch.fit_transform(redata.Parch.values.reshape(-1,1))

redata['Embarked'] = scale_emb.fit_transform(redata.Embarked.values.reshape(-1,1))

redata['Pclass'] = scale_class.fit_transform(redata.Pclass.values.reshape(-1,1))

redata['Sex'] = scale_sex.fit_transform(redata.Sex.values.reshape(-1,1))

redata.head()
# add dither to categoricals that have been converted to numerics, and to discrete numericals

redata['Pclass'] = redata['Pclass'] + np.random.randn(redata.shape[0])*0.01

redata['Sex'] = redata['Sex'] + np.random.randn(redata.shape[0])*0.1

redata['SibSp'] = redata['SibSp'] + np.random.randn(redata.shape[0])*0.1

redata['Parch'] = redata['Parch'] + np.random.randn(redata.shape[0])*0.1

redata['Embarked'] = redata['Embarked'] + np.random.randn(redata.shape[0])*0.1

redata['HasCabin'] = redata['HasCabin'] + np.random.randn(redata.shape[0])*0.01

redata['HasAgeRecorded'] = redata['HasAgeRecorded'] + np.random.randn(redata.shape[0])*0.01
redata.head()
correlations = redata.corr()

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

largeval = max(correlations.values.flatten()[correlations.values.flatten()<1])

smallval = min(correlations.values.flatten())

cax = ax.matshow(correlations, vmin=smallval, vmax=largeval)

fig.colorbar(cax)

ticks = np.arange(0,11,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(list(correlations),rotation=90)

ax.set_yticklabels(list(correlations))

plt.show()
from sklearn.linear_model import ARDRegression, LinearRegression



x_train = redata.as_matrix()[:,2:]

y_train = redata['Survived'].as_matrix()



clf = ARDRegression(compute_score=True)

clf.fit(x_train, y_train)

ols = LinearRegression()

ols.fit(x_train, y_train)
# accuracy of least-squares method



y_mean = clf.predict(x_train)

y_out = 1*(y_mean > 0.5)

y_ols_mean = ols.predict(x_train)

y_ols_out = 1*(y_ols_mean > 0.5)

print(sum(1*(y_out==y_train))/y_train.shape[0])

print(sum(1*(y_ols_out==y_train))/y_train.shape[0])
# xgboost method

import xgboost as xgb



gbm = xgb.XGBClassifier(max_depth=10, n_estimators=1000, learning_rate=0.5).fit(x_train, y_train)

predictions = gbm.predict(x_train)

accuracy = sum(1*(predictions == y_train)) / len(predictions)

print(accuracy)



#print(gbm.feature_importances_)

xgb.plot_importance(gbm)

plt.show()
test_data = pd.read_csv('../input/test.csv')



# wrangle test data the same way



test_data['HasCabin'] = 1.0*np.invert(test_data['Cabin'].isnull())

del test_data['Cabin']

del test_data['Ticket']

del test_data['Name']

test_data['HasAgeRecorded'] = 1.0*np.invert(test_data.Age.isnull())

test_data['Age'] = test_data['Age'].fillna(0)

test_data['Fare'] = test_data['Fare'].fillna(0)

test_data['Embarked'].replace(to_replace='S',value=1.,inplace=True)

test_data['Embarked'].replace(to_replace='C',value=2.,inplace=True)

test_data['Embarked'].replace(to_replace='Q',value=3.,inplace=True)

test_data['Embarked'].fillna(0,inplace=True)

test_data['Sex'].replace(to_replace='male',value=0.,inplace=True)

test_data['Sex'].replace(to_replace='female',value=1.,inplace=True)



test_data['Fare'] = scale_fare.transform(test_data.Fare.values.reshape(-1,1))

test_data['Age'] = test_data['Age']/np.max(test_data['Age'].values.reshape(-1,1))

test_data['SibSp'] = scale_sib.transform(test_data.SibSp.values.reshape(-1,1))

test_data['Parch'] = scale_parch.transform(test_data.Parch.values.reshape(-1,1))

test_data['Embarked'] = scale_emb.transform(test_data.Embarked.values.reshape(-1,1))

test_data['Pclass'] = scale_class.transform(test_data.Pclass.values.reshape(-1,1))

test_data['Sex'] = scale_sex.transform(test_data.Sex.values.reshape(-1,1))



test_data['Pclass'] = test_data['Pclass'] + np.random.randn(test_data.shape[0])*0.01

test_data['Sex'] = test_data['Sex'] + np.random.randn(test_data.shape[0])*0.1

test_data['SibSp'] = test_data['SibSp'] + np.random.randn(test_data.shape[0])*0.1

test_data['Parch'] = test_data['Parch'] + np.random.randn(test_data.shape[0])*0.1

test_data['Embarked'] = test_data['Embarked'] + np.random.randn(test_data.shape[0])*0.1

test_data['HasCabin'] = test_data['HasCabin'] + np.random.randn(test_data.shape[0])*0.01

test_data['HasAgeRecorded'] = test_data['HasAgeRecorded'] + np.random.randn(test_data.shape[0])*0.01



test_data.head()
#predictions_on_test = gbm.predict(test_data.as_matrix()[:,1:])

predictions_on_test = clf.predict(test_data.as_matrix()[:,1:])

predictions_on_test = 1*(predictions_on_test>0.5)

submission = pd.DataFrame({ 'PassengerId': test_data['PassengerId'],

                            'Survived': predictions_on_test })

submission.to_csv("submission.csv", index=False)
sum(1*(predictions_on_test==1))/len(predictions_on_test)
sum(predictions)/len(predictions)
predictions_on_test