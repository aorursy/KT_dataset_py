import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #for ploting 

#Import Linear REgression model from sklearn
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/gender_submission.csv')

print(df.head())
print(test.head())
print(sub.head())
CM1 = df[(df['Pclass']==1) & (df['Sex']=='male')]['Age'].mean()
CM2 = df[(df['Pclass']==2) & (df['Sex']=='male')]['Age'].mean()
CM3 = df[(df['Pclass']==3) & (df['Sex']=='male')]['Age'].mean()
CF1 = df[(df['Pclass']==1) & (df['Sex']=='female')]['Age'].mean()
CF2 = df[(df['Pclass']==2) & (df['Sex']=='female')]['Age'].mean()
CF3 = df[(df['Pclass']==3) & (df['Sex']=='female')]['Age'].mean()

M1 = test[(test['Pclass']==1) & (test['Sex']=='male')]['Age'].mean()
M2 = test[(test['Pclass']==2) & (test['Sex']=='male')]['Age'].mean()
M3 = test[(test['Pclass']==3) & (test['Sex']=='male')]['Age'].mean()
F1 = test[(test['Pclass']==1) & (test['Sex']=='female')]['Age'].mean()
F2 = test[(test['Pclass']==2) & (test['Sex']=='female')]['Age'].mean()
F3 = test[(test['Pclass']==3) & (test['Sex']=='female')]['Age'].mean()



print('\n',CM1,'\n',
      CM2,'\n',
      CM3,'\n',
      CF1,'\n',
      CF2,'\n',
      CF3)


df.loc[(df['Pclass']==1) & (df['Sex']=='male') & (df['Age'].isnull()),'Age'] = CM1
df.loc[(df['Pclass']==2) & (df['Sex']=='male') & (df['Age'].isnull()),'Age'] = CM2
df.loc[(df['Pclass']==3) & (df['Sex']=='male') & (df['Age'].isnull()),'Age'] = CM3
df.loc[(df['Pclass']==1) & (df['Sex']=='female') & (df['Age'].isnull()),'Age'] = CF1
df.loc[(df['Pclass']==2) & (df['Sex']=='female') & (df['Age'].isnull()),'Age'] = CF2
df.loc[(df['Pclass']==3) & (df['Sex']=='female') & (df['Age'].isnull()),'Age'] = CF3

test.loc[(test['Pclass']==1) & (df['Sex']=='male') & (test['Age'].isnull()),'Age'] = M1
test.loc[(test['Pclass']==2) & (df['Sex']=='male') & (test['Age'].isnull()),'Age'] = M2
test.loc[(test['Pclass']==3) & (df['Sex']=='male') & (test['Age'].isnull()),'Age'] = M3
test.loc[(test['Pclass']==1) & (df['Sex']=='female') & (test['Age'].isnull()),'Age'] = F1
test.loc[(test['Pclass']==2) & (df['Sex']=='female') & (test['Age'].isnull()),'Age'] = F2
test.loc[(test['Pclass']==3) & (df['Sex']=='female') & (test['Age'].isnull()),'Age'] = F3


df.info()
df.loc[(df['Sex']=='male'),'Sex'] = 0


test.loc[(test['Sex']=='male'),'Sex'] = 0
df.loc[(df['Sex']=='female'),'Sex'] = 1

test.loc[(test['Sex']=='female'),'Sex'] = 1
df.head()
df.loc[df['Embarked'] == 'S','Embarked'] = 1
df.loc[df['Embarked'] == 'C','Embarked'] = 2
df.loc[df['Embarked'] == 'Q','Embarked'] = 3
df.loc[df['Embarked'].isnull(),'Embarked'] = 0
test.loc[test['Embarked'] == 'S','Embarked'] = 1
test.loc[test['Embarked'] == 'C','Embarked'] = 2
test.loc[test['Embarked'] == 'Q','Embarked'] = 3
test.loc[test['Embarked'].isnull(),'Embarked'] = 0

df.info()
#Creating import and output variable X and Y
Xtrain = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Ytrain = df['Survived']

#Ytest = test[['Survived']]
fare3 = test[(test['Pclass']==3)]['Fare'].mean()

test.loc[(test['Pclass']==3) & (test['Fare'].isnull()),'Fare'] = fare3

print(fare3)


Xtest = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Xtest.info()
# Build and implement model and train it
lr = LinearRegression().fit(Xtrain,Ytrain)
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(Xtrain, Ytrain)
# Check the accuracy of our model lr
#lr.score(Xtrain,Ytrain)
clf.score(Xtrain,Ytrain)
from sklearn.linear_model import SGDClassifier
clf1 = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=2, eta0=2, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l1',
           power_t=0.5, random_state=None, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)

clf1.fit(Xtrain,Ytrain)
clf1.score(Xtrain,Ytrain)
#predict value of y but entering inpu

res = lr.predict(Xtest)
x = res.reshape(-1,1)
print(x)

ds = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':res})
mask1 = ds.Survived > 0.5
mask2 = ds.Survived < 0.5
ds.loc[mask1, 'Survived'] = int(1)

ds.loc[mask2, 'Survived'] = int(0)

ds['Survived']=pd.to_numeric(ds['Survived'], downcast='integer')
print(ds)


ds.to_csv('submission.csv', index=False)


        
    













