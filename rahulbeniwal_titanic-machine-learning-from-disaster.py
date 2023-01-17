# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

plt.rcParams["figure.figsize"] = (8, 4)

plt.rcParams["xtick.labelsize"] = 10

plt.rcParams["ytick.labelsize"] = 10



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_model = pd.read_csv('../input/gendermodel.csv')
train.shape,test.shape
train.head()
# columns

col = train.columns

col
train.drop(['PassengerId','Name','Ticket'],axis =1,inplace =True)

test_psg_id = test['PassengerId']

test.drop(['PassengerId','Name','Ticket'],axis =1,inplace =True)
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
train.Cabin.unique()
train.Cabin.str[0].unique() 
test.Cabin.str[0].unique()
train.Cabin = train.Cabin.str[0]

train.Cabin = train.Cabin.fillna("N")

sns.factorplot('Cabin','Survived', data=train,size=3,aspect=3)

#train = train.drop('Cabin',axis=1)

# for test

test.Cabin = train.Cabin.str[0]

test.Cabin = train.Cabin.fillna("N")
from sklearn import preprocessing

lc = preprocessing.LabelEncoder()

lc.fit(train['Cabin'])

train['Cabin']=lc.transform(train['Cabin'])

test['Cabin']=lc.transform(test['Cabin'])
missing_embark = train[train['Embarked'].isnull()]

missing_embark
similar_embark = train [(train['Fare']<82.0)&(train['Fare']>78.0)& (train['Cabin']==1)&(train['Pclass']==1)]

similar_embark
train.Embarked = train.Embarked.fillna('C')

sns.countplot(x='Embarked',hue ='Survived',data = train)

train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
lc = preprocessing.LabelEncoder()

lc.fit(train['Embarked'])

train['Embarked']=lc.transform(train['Embarked'])

test['Embarked']=lc.transform(test['Embarked'])
sns.factorplot('Pclass','Survived',data =train, size =3)
#lp = preprocessing.LabelEncoder()

#lp.fit(train['Pclass'])

#train['Pclass']=lp.transform(train['Pclass'])

#test['Pclass']=lp.transform(test['Pclass'])
train[['Fare','Survived']].groupby(['Survived'],as_index=False).mean()
df = train[['Fare','Survived']].groupby(['Fare'],as_index=False).mean()

sns.regplot(x=df.Fare,y= df.Survived,color="g")
df_fare = train[['Fare', 'Survived']]

df_fare['Fare'][df_fare['Fare']<300]=1

df_fare['Fare'][df_fare['Fare']>300]=2

sns.factorplot('Fare','Survived',data =df_fare, size =4)

df_fare.groupby(df_fare['Fare'],as_index=False).count()
#missing fare data in test

test[test['Fare'].isnull()]
# replacing missing fare vallue with the avg fare of passangers from First Class and Embarked from S

avg_fare=test[(test['Pclass'] == 3) & (test['Embarked'] == 2)]['Fare'].mean()

test["Fare"] = test["Fare"].fillna(avg_fare)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
ls = preprocessing.LabelEncoder()

ls.fit(train['Sex'])

train['Sex']=ls.transform(train['Sex'])

test['Sex']=ls.transform(test['Sex'])
train.columns
sns.countplot(x='SibSp',hue ='Survived',data = train)

train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean()
sns.countplot(x='Parch',hue ='Survived',data = train)

train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
age_null_count = train.Age.isnull().sum()

age_avg = train.Age.mean()

age_std = train.Age.std()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

train['Age'][np.isnan(train['Age'])] = age_null_random_list

train['Age'] = train['Age'].astype(int)

#for test

test_age_null_count = test.Age.isnull().sum()

test_age_avg = test.Age.mean()

test_age_std = test.Age.std()

test_age_null_random_list = np.random.randint(test_age_avg - test_age_std, test_age_avg + test_age_std, size=test_age_null_count)

test['Age'][np.isnan(test['Age'])] = test_age_null_random_list

test['Age'] = test['Age'].astype(int)



train['CategoricalAge'] = pd.cut(train['Age'], 5)

train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
train = train.drop('CategoricalAge',1)
test.Age.isnull().sum()
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(train[['Age', 'Fare']])

train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])





#std_scale = preprocessing.StandardScaler().fit(test[['Age', 'Fare']])

test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
y = train.Survived.as_matrix()

X = train.drop('Survived',1).as_matrix()

#test_psg_id = test['PassengerId']

X_test = test.as_matrix()
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(

    X, y, test_size=0.2, random_state=42)
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import  metrics



classifiers=[RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),

             SVC(),LogisticRegression(),DecisionTreeClassifier(),GaussianNB(),MLPClassifier()]

#accuracy = {}

pr = {}

for cl in classifiers:

    clf = cl

    clf.fit(X,y)

    predicted = clf.predict(X_test)

    pr[clf.__class__.__name__] = predicted

    #acc = metrics.accuracy_score(y_cv, predicted)

    #accuracy[clf.__class__.__name__]=acc
#accuracy_df = pd.DataFrame.from_dict(accuracy,orient ='index',)

#accuracy_df
pr_df = pd.DataFrame.from_dict(pr,orient = 'columns')

pr_df
pr_df.shape
pr_df = pr_df.iloc[:,3:8]

pr_df.head()
y_pred = pr_df.sum(axis=1)

y_pred[y_pred>2]=1

y_pred[y_pred!=1]=0
test_pred = y_pred
#vot_acc = metrics.accuracy_score(y_cv, y_pred)

#vot_acc
sub1 = pd.DataFrame(test_psg_id,columns =['PassengerId'])

sub2 = pd.DataFrame(test_pred,columns = ['Survived'])

output = pd.concat([sub1,sub2],axis =1)

output.to_csv('vt_submission.csv', index=False)

output.head(10)
