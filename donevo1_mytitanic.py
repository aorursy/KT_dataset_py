import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.width', 200)

pd.set_option('display.max_columns', None)

pd.options.display.float_format = '{:11,.2f}'.format
df_train = pd.read_csv('../input/titanic/train.csv',index_col=0)

df_train.head(2)
df_test = pd.read_csv('../input/titanic/test.csv',index_col=0)

df_test.head(2)
df_combined = pd.concat([df_train,df_test])
df_combined.info()
title = df_combined.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

title.value_counts()
title_dict = {'Mr':'Mr',  'Mrs':'Mrs',  'Miss':'Miss',  'Master':'Master',  'Don':'Mr',  'Rev':'Mr',  'Dr':'Mr','Mme':'Mrs', 'Ms':'Miss', 

              'Major':'Mr', 'Lady':'Mrs', 'Sir':'Mr', 'Mlle':'Miss', 'Col':'Mr', 'Capt':'Mr', 'the Countess':'Mrs', 'Jonkheer':'Mr', 'Dona':'Mrs'}

df_combined['Title'] = title.map(title_dict)
df_combined.groupby(['Title','Pclass'])['Age'].mean()
df_combined['Age'] = df_combined['Age'].fillna(df_combined.groupby(['Title', 'Pclass'])['Age'].transform('mean'))
print(df_combined['Embarked'].value_counts())

df_combined.Embarked.fillna(value='S',inplace=True) # replace by 'S' since it is the most frequent
sns.kdeplot(df_combined.loc[(df_combined['Pclass']==1), 'Age'], color='r', shade=True, Label='Class=1')  

sns.kdeplot(df_combined.loc[(df_combined['Pclass']==2), 'Age'], color='b', shade=True, Label='Class=2') 

sns.kdeplot(df_combined.loc[(df_combined['Pclass']==3), 'Age'], color='g', shade=True, Label='Class=3') 

plt.xlabel('Age');

plt.ylabel('Probability Density');
plt.hist(df_combined['Age'], color = 'blue', edgecolor = 'black', bins = int(20));
df_combined['age_group'] = pd.cut(df_combined['Age'], bins=[i for i in range(0,81,5)], right=False) #,labels=[i for i in range(1,81,5)])

df_combined[df_combined['Sex']=='male'].groupby('age_group').agg({'Survived': ['count', 'sum', 'mean']}).head(5)
df_combined['is_child'] = np.where(df_combined['Age'] < 10, 1, 0)
df_combined['family_size'] = df_combined.SibSp + df_combined.Parch

df_combined[df_combined['Sex']=='male'].groupby('family_size').agg({'Survived': ['count', 'sum', 'mean']})
df_combined.describe(include='all')
df_combined.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Fare','Cabin', 'Title', 'Age', 'age_group'],inplace=True)

df_combined.head()
combined = pd.get_dummies(df_combined, columns=['Pclass','Sex', 'Embarked', 'family_size'], drop_first=True)
combined.head()
train = combined[:df_train.shape[0]].copy()

test = combined[df_train.shape[0]:].copy()

test.drop('Survived',axis=1,inplace=True)
def split_data(df_train):

  _df_train, df_valid = train_test_split(df_train, test_size=0.6, random_state=12345)

  features_train = _df_train.drop(['Survived'], axis=1)

  target_train = _df_train['Survived']

  features_valid = df_valid.drop(['Survived'], axis=1)

  target_valid = df_valid['Survived']

  return features_train, target_train, features_valid, target_valid
features_train, target_train, features_valid, target_valid = split_data(train)
features_train.describe(include='all')
model = LogisticRegression(random_state=12345, class_weight=None)

model.fit(features_train, target_train)

predictions_valid = model.predict(features_valid)

print("LogisticRegression:", accuracy_score(target_valid, predictions_valid))
def learning_curve(train, valid, steps, target):

    plt.figure(figsize=(9, 9))

    targets = [target]* len(train)

    plt.plot(steps, train, 'o-', color="r", label="Training")

    plt.plot(steps, valid, 'o-', color="b", label="Validation")

    plt.plot(steps, targets,'-', color="g", label="Target")

    plt.ylabel('Score') 

    plt.title('Learning Curve')

    plt.legend()

    plt.show()
score_train = []

score_valid = []

steps = []

for depth in range(2,20):

    model = DecisionTreeClassifier(random_state=12345, max_depth=depth, class_weight='balanced')

    model.fit(features_train, target_train)

    steps.append(depth)

    score_train.append(accuracy_score(target_train, model.predict(features_train)))

    score_valid.append(accuracy_score(target_valid, model.predict(features_valid)))

learning_curve(score_train, score_valid, steps, 0.59)
score_train = []

score_valid = []

steps = []

for depth in range(2,10):

    model = RandomForestClassifier(random_state=12345, max_depth=depth, class_weight='balanced')

    model.fit(features_train, target_train)

    steps.append(depth)

    score_train.append(accuracy_score(target_train, model.predict(features_train)))

    score_valid.append(accuracy_score(target_valid, model.predict(features_valid)))



learning_curve(score_train, score_valid, steps, 0.59)
model = RandomForestClassifier(random_state=12345, max_depth=5, class_weight='balanced')

model.fit(features_train, target_train)

print(

    accuracy_score(target_train, model.predict(features_train)),

    accuracy_score(target_valid, model.predict(features_valid)),

    )
predicted_test = model.predict(test)

predicted_test = predicted_test.astype(int)



submission = pd.DataFrame({

        "PassengerId": test.index,

        "Survived": predicted_test

    })



submission.to_csv("titanic_submission.csv", index=False)
!ls