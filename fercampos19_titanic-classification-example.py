import pandas as pd

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt 

import seaborn as sns



%matplotlib inline
## for exploration we'll use only training data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train.shape)

print(test.shape)
train.info()
train.describe()
## divide quantitive and categorical variables

df_num = train[['Age','SibSp','Parch','Fare']]

df_cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
#distributions for all quantitive variables 

for i in df_num.columns:

    plt.hist(df_num[i], bins=10)

    plt.title(i)

    plt.show();
corr = df_num.corr()

print(corr)

sns.heatmap(corr, cmap='YlGnBu');
sns.pairplot(df_num);
sns.violinplot(x=train.Survived, y=train.Age, data=train);
sns.violinplot(x=train.Survived, y=train.Fare, data=train);
sns.violinplot(x=train.Survived, y=train.SibSp, data=train);
sns.violinplot(x=train.Survived, y=train.Parch, data=train);
df_num = train[['Age','SibSp','Parch','Fare', 'Survived']]



pd.pivot_table(df_num, index='Survived', aggfunc=['mean', 'median'])
for i in df_cat.columns:

    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)

    plt.show()
for i in df_cat.columns:

    if i in ['Survived', 'Cabin', 'Ticket']:

        pass

    else:

        ct = pd.crosstab(train[i], train.Survived).apply(lambda x: x/x.sum(), axis=1)

        ct.plot(kind='bar', stacked=True)

for i in df_cat.columns:

    if i in ['Survived', 'Cabin', 'Ticket']:

        pass

    else:

        ct = pd.crosstab(train.Survived, train[i]).apply(lambda x: x/x.sum(), axis=1)

        ct.plot(kind='bar', stacked=True)
## making age groups

train['Age'] = train.Age.fillna(train.Age.median())

train['agegroup'] = pd.cut(train.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80], include_lowest=False)

train.agegroup.value_counts()
## see how agegroup relate to Survived

pd.crosstab(train.Survived, train.agegroup).apply(lambda x: x/x.sum(), axis=0)
## getting person's title

train['name_title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

train.name_title.value_counts()
## see how name title relate to Survived

pd.crosstab(train.Survived, train.name_title).apply(lambda x: x/x.sum(), axis=0)
## create new categorical in test

test['Age'] = test.Age.fillna(test.Age.median())

test['agegroup'] = pd.cut(test.Age, [0, 10, 20, 30, 40, 50, 60, 70, 80], include_lowest=False)

test['name_title'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())



print(train.columns)

print(test.columns)
train.isna().value_counts().unstack()
test.isna().value_counts().unstack()
train.dropna(subset=['Embarked'], inplace=True)

test['Fare'] = test.Fare.fillna(test.Fare.median())
train.isna().value_counts().unstack()
test.isna().value_counts().unstack()
import numpy as np



## normalize fare

train['Fare'] = np.log(train.Fare+1)

test['Fare'] = np.log(test.Fare+1)



print(train.columns)

print(test.columns)
## Select features and target

features = ['Pclass', 'Sex', 'Embarked', 'agegroup', 'SibSp', 'Parch', 'Fare', 'name_title']



y = train['Survived']

X_train = train[features]

X_test = test[features]



print(X_train.columns)

print(X_test.columns)
## get dummies

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)
# Scale numeric data data

from sklearn.preprocessing import StandardScaler

X_train[['SibSp','Parch','Fare']] = StandardScaler().fit_transform(X_train[['SibSp','Parch','Fare']])

X_test[['SibSp','Parch','Fare']] = StandardScaler().fit_transform(X_test[['SibSp','Parch','Fare']])
print(X_train.shape)

print(X_test.shape)
X_train.drop(X_train.columns.difference(list(X_test.columns)), axis=1, inplace=True) 

X_test.drop(X_test.columns.difference(list(X_train.columns)), axis=1, inplace=True)

print(X_train.shape)

print(X_test.shape)
print(list(X_test.columns))

print(list(X_train.columns))
## import cross_val for the scores

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

cv = cross_val_score(lr, X_train, y, cv=5)

print(cv)

print(cv.mean())
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

cv = cross_val_score(gnb, X_train, y, cv=5)

print(cv)

print(cv.mean())
from sklearn.svm import SVC



# The default SVC kernel is radial basis function (RBF)

svc_rbf = SVC(probability=True)

cv = cross_val_score(svc_rbf, X_train, y, cv=5)

print('RBF kernel')

print(cv)

print(cv.mean())

print()



# kernel poly

svc_poly = SVC(kernel='poly', probability=True)

cv = cross_val_score(svc_poly, X_train, y, cv=5)

print('Poly kernel')

print(cv)

print(cv.mean())
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_predict



error = []

for k in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=k)

    y_pred = cross_val_predict(knn, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(1,51), error);
## look only between 1 and 10



error = []

for k in range(1,11):

    knn = KNeighborsClassifier(n_neighbors=k)

    y_pred = cross_val_predict(knn, X_train, y, cv=5)

    error.append(mean_squared_error(y,y_pred))

    

plt.plot(range(1,11), error);
knn = KNeighborsClassifier(n_neighbors=7)

cv = cross_val_score(knn, X_train, y, cv=5)

print(cv)

print(cv.mean())
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100)

cv = cross_val_score(rf, X_train, y, cv=5)

print(cv)

print(cv.mean())
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(n_estimators=150, random_state=0)

cv = cross_val_score(gbc, X_train, y, cv=5)

print(cv)

print(cv.mean())
from sklearn.ensemble import VotingClassifier

voting_clf_soft = VotingClassifier(estimators = [('lr',lr),('gnb',gnb),('svc_rbf',svc_rbf),('svc_poly',svc_poly), 

                                                 ('knn',knn),('rf',rf),('gbc',gbc)], 

                                   voting = 'soft')

voting_clf_hard = VotingClassifier(estimators = [('lr',lr),('gnb',gnb),('svc_rbf',svc_rbf),('svc_poly',svc_poly),

                                                 ('knn',knn),('rf',rf),('gbc',gbc)], 

                                   voting = 'hard')
cv = cross_val_score(voting_clf_soft,X_train,y,cv=5)

print('soft')

print(cv)

print(cv.mean())

print()



cv = cross_val_score(voting_clf_hard,X_train,y,cv=5)

print('hard')

print(cv)

print(cv.mean())
voting_clf_soft.fit(X_train,y)

predictions = voting_clf_soft.predict(X_test)
## make the file to submit



df_submition = pd.read_csv("/kaggle/input/titanic/test.csv")



df_submition['Survived'] = predictions



df_submition.drop(df_submition.columns.difference(['PassengerId', 'Survived']), axis=1, inplace=True) # Selecting only needed columns



df_submition.head(5) 
## file is expected to have 418 rows



df_submition.count()
df_submition.to_csv('my_submission.csv', index=False)

print('File created')