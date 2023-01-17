import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.info()
train_df.head(25)
test_df.info()
median_age = train_df['Age'].median()
train_df['Age'] = train_df['Age'].fillna(median_age)
test_df['Age'] = test_df['Age'].fillna(median_age)

most_frequent_Embarked = train_df['Embarked'].value_counts().index[0]
train_df['Embarked'] = train_df['Embarked'].fillna(most_frequent_Embarked)
test_df['Embarked'] = test_df['Embarked'].fillna(most_frequent_Embarked)

train_df['Cabin'] = train_df['Cabin'].fillna('!!!')
train_df['Cabin'] =  train_df['Cabin'].apply(lambda x: 0 if x == '!!!' else 1)

test_df['Cabin'] = test_df['Cabin'].fillna('!!!')
test_df['Cabin'] =  test_df['Cabin'].apply(lambda x: 0 if x == '!!!' else 1)

train_df['Sex'] = train_df['Sex'] == 'male'
train_df['Sex'] = train_df['Sex'].astype(int)

test_df['Sex'] = test_df['Sex'] == 'male'
test_df['Sex'] = test_df['Sex'].astype(int)

train_df['Family_size'] = train_df['SibSp'] + train_df['Parch']
test_df['Family_size'] = test_df['SibSp'] + test_df['Parch']

train_df['Alone'] = train_df['Family_size'] == 0
train_df['Alone'] = train_df['Alone'].astype(int)

test_df['Alone'] = test_df['Family_size'] == 0
test_df['Alone'] = test_df['Alone'].astype(int)

train_df['Title'] = [x.split(', ')[1].split('. ')[0] for x in train_df['Name']]
most_frequent_titles = ['Mr','Miss','Mrs','Master']
train_df['Title'] = train_df['Title'].apply(lambda x: x if x in most_frequent_titles else 'Rare')

test_df['Title'] = [x.split(', ')[1].split('. ')[0] for x in test_df['Name']]
test_df['Title'] = test_df['Title'].apply(lambda x:x if x in most_frequent_titles else 'Rare')
 
train_df["t_has_prefix"] = train_df['Ticket'].apply(lambda x: 1 if len(x.split())>1 else 0)
test_df["t_has_prefix"] = test_df['Ticket'].apply(lambda x: 1 if len(x.split())>1 else 0)

test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].mean())
train_df_dummies = pd.get_dummies(train_df,columns = ['Title','Embarked'])
test_df_dummies = pd.get_dummies(test_df,columns = ['Title','Embarked'])

train_df, test_df = train_df_dummies.align(test_df_dummies,join='left',axis=1)
sns_plot = sns.distplot(train_df['Fare'])
plt.show()
train_df = train_df[train_df['Fare']<200]
sns_plot = sns.distplot(train_df['Fare'])
plt.show()
sns_plot = sns.distplot(train_df['SibSp'])
plt.show()
train_df = train_df[train_df['SibSp']<8]
sns_plot = sns.distplot(train_df['SibSp'])
plt.show()
test_PassengerId = test_df['PassengerId']
target = train_df['Survived']

train_df = train_df.drop(['Name','Ticket','Survived','PassengerId'],axis=1)
test_df = test_df.drop(['Name','Ticket','Survived','PassengerId'],axis=1)
train_df.info()
test_df.info()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

sc = StandardScaler()
X = sc.fit_transform(train_df)
y = target

classifier = DecisionTreeClassifier(max_depth=5)

scores = cross_val_score(classifier, X, y,cv=5)
res = scores.mean()

print('DecisionTreeClassifier result: {}\nScores: {}'.format(res,scores))
from sklearn.feature_selection import SelectPercentile

select = SelectPercentile(percentile=70)
select.fit(X, y)
X = select.transform(X)

scores = cross_val_score(classifier, X, y,cv=5)
res = scores.mean()

print('DecisionTreeClassifier result(after feature selection): {}\nScores: {}'.format(res,scores))
classifier.fit(X,y)

print('Feature importance ranking\n\n')
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

importance_list = []
for f in range(X.shape[1]):
    variable = train_df.columns[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(importance_list, importances[indices],
       color="r", align="center")
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile

sc = StandardScaler()
X = sc.fit_transform(train_df)

select = SelectPercentile(percentile=70)
select.fit(X, y)
X = select.transform(X)

y = target

GBC_classifier = GradientBoostingClassifier(random_state=5,n_estimators=3000,learning_rate=0.008)
SVC_classifier = SVC(C=0.5,gamma = 0.07,probability=True)
classifier = VotingClassifier(estimators=[('GBC_classifier', GBC_classifier), ('SVC_classifier', SVC_classifier)],voting='soft') 

scores = cross_val_score(classifier, X, y,cv=5)
res = scores.mean()

print('VotingClassifier evaluating result: {}\nScores: {}'.format(res,scores))
classifier.fit(X,y)
X = test_df
X = sc.transform(X)
X = select.transform(X)

y = classifier.predict(X)

test_df['Survived'] = y

test_df.head(15)

res_df = pd.DataFrame(data={'PassengerId':test_PassengerId,'Survived':y})

res_df.to_csv('res.csv',index = False)