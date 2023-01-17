!pwd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.read_csv('../input/titanic/train.csv')#../は一階層上のディレクトリを指す

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

data = pd.concat([train,test],sort=False)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Fare'].fillna(np.mean(data['Fare']),inplace=True)
#data.isnull().sum()

data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

X_train.head()
y_train.head()


from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(penalty='l2',solver='sag',random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(len(y_pred))
sub = pd.read_csv('../input/titanic/gender_submission.csv')
print(len(sub))
sub['Survived'] = list(map(int, y_pred))
sub.to_csv('submission.csv', index=False)
import pandas as pd
import pandas_profiling


train = pd.read_csv('../input/titanic/train.csv')
train.profile_report()
import matplotlib.pyplot as plt
plt.hist(train.loc[train['Survived']== 0,'Age'].dropna(),
        bins=40,alpha=0.5,label='0')
plt.hist(train.loc[train['Survived']==1,'Age'].dropna(),
        bins=40,alpha=0.5,label='1')
plt.xlabel('Age')
plt.ylabel('count')
plt.legend(title='Survived')
import seaborn as sns
sns.countplot(x='SibSp',hue='Survived',data=train)
plt.legend(loc='upper right',title='Survived')
import seaborn as sns

x_min = int(train["Age"].min())

# 最大値
x_max = int(train["Age"].max())

# 最小値から最大値の範囲で5間隔
range_bin_width = range(x_min, x_max, 5)



sns.countplot(x='Age',hue='Survived',data=train)
plt.legend(loc='upper right',title='Survived')
