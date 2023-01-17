import numpy as np 

import pandas as pd 



import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train['type_set'] = 'train'

test['type_set'] = 'test'

sets = pd.concat([train,test],sort=False)

print('train.shape:', train.shape)

print('test.shape:', test.shape)

print('sets.shape:', sets.shape)

sets.head()
sets.info()
sets[sets.Embarked.isnull()==True]
sets.Embarked.value_counts()
sets.Embarked.fillna('S', inplace = True)

sets[sets.Embarked.isnull()==True]
sets[sets.Fare.isnull()==True]
sets[sets.Pclass == 3].mean()['Fare']
sets.Fare.fillna(13.30, inplace = True)
sets.Cabin.fillna('N', inplace = True)

sets.Cabin.unique()
sets['Cabin_type'] = [i[0] for i in sets.Cabin]

sets.Cabin_type.unique()
a_table = sets[['Survived','Cabin_type']].groupby(['Cabin_type']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
def feature(f):

    if f == 'N': return 0

    else: return 1

sets['Has_cabin'] = sets['Cabin_type'].apply(lambda x: feature(x))



a_table = sets[['Survived','Has_cabin']].groupby(['Has_cabin']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
def feature(f):

    if f < 8: return 'infant'

    #if f < 18: return 'child'

    if f < 24: return 'young'

    #if f < 45: return 'Adult'

    if f < 60: return 'senior'

    if f >= 60: return 'old'    



sets['Age_group'] = sets['Age'].apply(lambda x: feature(x))



a_table = sets[['Survived','Age_group']].groupby(['Age_group']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
sets["title"] = [i.split('.')[0] for i in sets.Name]

sets["title"] = [i.split(', ')[1] for i in sets.title]
sets["title"].value_counts()
av_age = sets[['title','Age']].groupby(['title']).agg(['mean', 'count'])

av_age.columns = av_age.columns.droplevel(0)

av_age.reset_index(inplace = True)

av_age.sort_values(['count'], ascending=False)
nul_age = sets[(sets.Age.isnull()==True)]



a_table = nul_age[['title','PassengerId']].groupby(['title']).agg(['count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.reset_index(inplace = True)

a_table.sort_values(['count'], ascending=False)
for i in list(a_table.title.unique()):

    m1 = (sets['title'] == i) & (sets['Age'].isnull()==True)

    sets.loc[m1,'Age'] = sets.loc[m1,'Age'].fillna(round(av_age[av_age.title == i]['mean'].iloc[0]))
sets['Age_group'] = sets['Age'].apply(lambda x: feature(x))
sets.info()
a_table = sets[['Survived','Pclass']].groupby(['Pclass']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
sets['len_name'] = [len(i) for i in sets.Name]



def feature(x):

    if (x < 20): return 'short'

    if (x < 27): return 'medium'

    if (x < 32): return 'good'

    else: return 'long'



sets['len_name_g'] = sets['len_name'].apply(lambda x: feature(x))





a_table = sets[['Survived','len_name_g']].groupby(['len_name_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
a_table = sets[['Survived','Sex']].groupby(['Sex']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
a_table = sets[['Survived','SibSp']].groupby(['SibSp']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
def feature(x):

    if (x < 1): return '0'

    if (x < 2): return '1'

    #if (x < 32): return '1+'

    else: return '1+'



sets['SibSp_g'] = sets['SibSp'].apply(lambda x: feature(x))



a_table = sets[['Survived','SibSp_g']].groupby(['SibSp_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
a_table = sets[['Survived','Parch']].groupby(['Parch']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
def feature(x):

    if (x < 1): return '0'

    else: return '1+'



sets['Parch_g'] = sets['Parch'].apply(lambda x: feature(x))



a_table = sets[['Survived','Parch_g']].groupby(['Parch_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
#sets.Ticket.unique()
sets['len_Ticket'] = [len(i) for i in sets.Ticket]



a_table = sets[['Survived','len_Ticket']].groupby(['len_Ticket']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['len_Ticket'], ascending=False)
def feature(x):

    if (x == 5 or x == 5): return '5and8'

    else: return 'other'



sets['len_Ticket_g'] = sets['len_Ticket'].apply(lambda x: feature(x))





a_table = sets[['Survived','len_Ticket_g']].groupby(['len_Ticket_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
sets['Ticket_type'] = [i.split()[0] for i in sets.Ticket]

sets['Ticket_type'].value_counts()



a_table = sets[['Survived','Ticket_type']].groupby(['Ticket_type']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['count'], ascending=False).head(10)
def feature(x):

    if (x == 'PC'): return 'PC'

    else: return 'other'



sets['Ticket_type_g'] = sets['Ticket_type'].apply(lambda x: feature(x))





a_table = sets[['Survived','Ticket_type_g']].groupby(['Ticket_type_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
def feature(x):

    if (x < 10): return 'under10'

    if (x < 50): return '10-50'

    else: return '50+'



sets['Fare_g'] = sets['Fare'].apply(lambda x: feature(x))





a_table = sets[['Survived','Fare_g']].groupby(['Fare_g']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
a_table = sets[['Survived','Embarked']].groupby(['Embarked']).agg(['mean', 'count'])

a_table.columns = a_table.columns.droplevel(0)

a_table.sort_values(['mean'], ascending=False)
list(sets)
train_set = sets[sets.type_set == 'train']

train_set.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

print(train_set.shape)

train_set.head() 
list(train_set)
train_set = pd.get_dummies(train_set, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)
train_set.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.model_selection import train_test_split



y = train_set["Survived"]

X = train_set.drop(['Survived'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)
logreg = LogisticRegression()



# fit the model with "train_x" and "train_y"

logreg.fit(X_train,y_train)



y_pred = logreg.predict(X_test)



round(accuracy_score(y_pred, y_test),4)
test_set = sets[sets.type_set == 'test']

test_set.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

test_set = pd.get_dummies(test_set, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)
logreg.fit(X,y)



#predict test set

set_pred = logreg.predict_proba(test_set)[:, 1]
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':set_pred})

submission.to_csv('submission.csv')
sets_W_I = sets.loc[(sets.Sex == 'female') | (sets.Age_group == 'infant')]

sets_Otr = sets.loc[(sets.Sex != 'female') & (sets.Age_group != 'infant')]
sets_W_I["Survived"].mean()
sets_Otr["Survived"].mean()
#sets_W_I

train_set_W_I = sets_W_I[sets_W_I.type_set == 'train']

train_set_W_I.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)



train_set_W_I = pd.get_dummies(train_set_W_I, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)



y = train_set_W_I["Survived"]

X = train_set_W_I.drop(['Survived'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)



logreg_W_I = LogisticRegression()



# fit the model with "train_x" and "train_y"

logreg_W_I.fit(X_train,y_train)



y_pred = logreg_W_I.predict(X_test)



round(accuracy_score(y_pred, y_test),4)
logreg_W_I.fit(X,y)
#sets_Otr

train_set_Otr = sets_Otr[sets_Otr.type_set == 'train']

train_set_Otr.drop(['PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)



train_set_Otr = pd.get_dummies(train_set_Otr, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)



y = train_set_Otr["Survived"]

X = train_set_Otr.drop(['Survived'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=0)



logreg_Otr = LogisticRegression()



# fit the model with "train_x" and "train_y"

logreg_Otr.fit(X_train,y_train)



y_pred = logreg_Otr.predict(X_test)



round(accuracy_score(y_pred, y_test),4)
logreg_Otr.fit(X,y)
#sets_W_I

test_set_W_I = sets_W_I[sets_W_I.type_set == 'test']

test_set_W_I.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

test_set_W_I = pd.get_dummies(test_set_W_I, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)



#sets_Otr

test_set_Otr = sets_Otr[sets_Otr.type_set == 'test']

test_set_Otr.drop(['Survived','PassengerId','Name','Age','SibSp', 'Parch','Ticket', 'Fare', 'Cabin','type_set','Cabin_type',

               'title','len_name','len_Ticket','Ticket_type', ], axis=1, inplace=True)

test_set_Otr = pd.get_dummies(test_set_Otr, columns=[ 'Pclass',

 'Sex',

 'Embarked',

 'Has_cabin',

 'Age_group',

 'len_name_g',

 'SibSp_g',

 'Parch_g',

 'len_Ticket_g',

 'Ticket_type_g',

 'Fare_g'], drop_first=True)





#predict test sets

set_pred_W_I = logreg_W_I.predict_proba(test_set_W_I)[:, 1]

set_pred_Otr = logreg_Otr.predict_proba(test_set_Otr)[:, 1]
submission_W_I = pd.DataFrame({'PassengerId': sets_W_I[sets_W_I.type_set == 'test']['PassengerId'], 'Survived':set_pred_W_I})

submission_Otr = pd.DataFrame({'PassengerId': sets_Otr[sets_Otr.type_set == 'test']['PassengerId'], 'Survived':set_pred_Otr})



submission_2models = pd.concat([submission_W_I, submission_Otr], ignore_index=True)



submission_2models.to_csv('submission_2models.csv')