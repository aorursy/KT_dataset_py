import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = train['Survived']  #target variable
train = train.drop('Survived', axis=1)
train['training_set'] = True
test['training_set'] = False
full = pd.concat([train, test])
full = full.interpolate()
full = pd.get_dummies(full)
train = full[full['training_set']==True]
train = train.drop('training_set', axis=1)
test = full[full['training_set']==False]
test =test.drop('training_set', axis=1)
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train, target)
preds = rf.predict(test)
my_submission = pd.DataFrame({'PassengerId': test.index, 'Survived': preds})
my_submission['Survived']=my_submission['Survived'].apply(lambda x: 1 if x > 0 else 0)
my_submission.to_csv('submission.csv', index=False)
