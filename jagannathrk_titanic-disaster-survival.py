import pandas as pd

train = pd.read_csv('/kaggle/input/titanic/train.csv').set_index('PassengerId')

test = pd.read_csv('/kaggle/input/titanic/test.csv').set_index('PassengerId')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



titanic = pd.concat([train,test], axis=0, sort=False) #Concat on the row axis

display(titanic.describe(include='all'))

titanic.isnull().sum()
titanic['Cabin_Missing'] = titanic.Cabin.isnull()*1.0

titanic.drop(['Cabin'], axis=1, inplace=True)
titanic['Title'] = titanic.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

titanic['LastName'] = titanic.Name.str.split(',').str[0]

titanic.drop(['Name'], axis=1 ,inplace=True)

titanic.LastName, _ = pd.factorize(titanic['LastName'])
alone_child = titanic.query('Age < 12 & Parch == 0').index

for i in alone_child:

    titanic.at[i,'Parch'] += 1 #add nanny
def get_ticket_type(ticket):

  ticket_list = ticket.split(' ')

  if len(ticket_list) == 2:

    return ticket_list[0]

  else:

    ticket_list = ticket.split('.')

    return 'Normal'



def get_ticket_number(ticket):

  ticket_list = ticket.split(' ')

  if len(ticket_list) == 2:

    return ticket_list[1]

  else:

    return ticket



titanic['Ticket_Type'] = titanic['Ticket'].apply(lambda x: get_ticket_type(x))

titanic['Ticket_Number'] = titanic['Ticket'].apply(lambda x: get_ticket_number(x))

missing_index = titanic[pd.to_numeric(titanic['Ticket_Number'], errors='coerce').isnull()].index

missing_index = missing_index.to_list()

df_ticket_type_correction = ['STON/O2.','STON/O2.','LINE','STON/O2.','LINE','LINE','STON/O2.','STON/O2.','STON/O2.','STON/O2.','SC/AH Basle','STON/O2.','STON/O2.','LINE','STON/O2.',

                             'STON/O2.', 'STON/O2.', 'STON/O2.', 'STON/O2.', 'A./2.']

df_ticket_number_correction = [3101294,2101280,0,3101275,0,0,3101293,3101289,3101269,3101274,541,3101286,3101273,0,3101292,3101285,3101288,3101291,3101268,39186]

for i in range(20):

  titanic.at[missing_index[i], 'Ticket_Type'] = df_ticket_type_correction[i]

  titanic.at[missing_index[i], 'Ticket_Number'] = df_ticket_number_correction[i]

titanic['Ticket_Number'] = pd.to_numeric(titanic['Ticket_Number'])

titanic['Ticket_Type'] = titanic['Ticket_Type'].apply(lambda x: x.replace('.','').upper())

titanic.drop(['Ticket'], axis=1, inplace=True)

titanic.drop(['Ticket_Type'], axis=1, inplace=True)

titanic.Ticket_Number, _ = pd.factorize(titanic['Ticket_Number'])
titanic.isnull().sum()
print(titanic.Embarked.value_counts())

titanic.Embarked = titanic.Embarked.fillna('S')
age_mean_dict = titanic.groupby(['Pclass','Title','Embarked'])['Age'].mean().to_dict()

age_mean_dict[(3,'Ms','Q')] = titanic.groupby(['Title'])['Age'].mean().to_dict()['Ms']

print(age_mean_dict)

missing_age_index = titanic[titanic.Age.isnull()].index

for i in missing_age_index:

    titanic.at[i,'Age'] = age_mean_dict[(titanic.at[i,'Pclass'], titanic.at[i,'Title'], titanic.at[i,'Embarked'])]
missing_fare_index = titanic[titanic.Fare.isnull()].index

titanic.at[missing_fare_index, 'Fare'] = titanic.groupby(['Pclass','Embarked','Title'])['Age'].mean()[(3,'S','Mr')]
titanic['WomanOrChild'] = ((titanic.Title == 'Master') | (titanic.Sex == 'female'))
titanic.Pclass = titanic.Pclass.apply(lambda x: ['Dummy','Rich','Middle','Poor'][x])

titanic['Family_Size'] = titanic['SibSp']+titanic['Parch']
titanic.loc[titanic.Survived.isnull(),'Survived'] = titanic.loc[titanic.Survived.isnull(),'WomanOrChild'] * 1.0

family = titanic.groupby(['LastName']).Survived

friends = titanic.groupby(['Ticket_Number']).Survived
titanic['WomanOrBoyCount'] = family.transform(lambda s: s[titanic.WomanOrChild].fillna(0).count())

titanic['WomanOrBoyCount'] = titanic.mask(titanic.WomanOrChild, titanic.WomanOrBoyCount - 1, axis=0)
titanic['WomanOrBoyCount2'] = friends.transform(lambda s: s[titanic.WomanOrChild].fillna(0).count())

titanic['WomanOrBoyCount2'] = titanic.mask(titanic.WomanOrChild, titanic.WomanOrBoyCount2 - 1, axis=0)

titanic.WomanOrChild = titanic.WomanOrChild * 1.0
titanic['Alone'] = (titanic.Family_Size == 0) * 1.0
titanic.head()
titanic_onehot = pd.get_dummies(titanic, columns=['Pclass','Sex','Title','Embarked'])
train_cols = [col.replace('(','_').replace(']','_').replace('<','_') for col in titanic_onehot.columns]

titanic_onehot.columns = train_cols
train_onehot = titanic_onehot.loc[train.index]

test_onehot = titanic_onehot.loc[test.index]

assert train_onehot['Survived'].isnull().sum() == 0

test_onehot.drop(['Survived'], axis=1, inplace=True)
!pip install rfpimp
from rfpimp import *

train_X = train_onehot.drop(['Survived'], axis=1)

train_y = train_onehot['Survived']

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf.fit(train_X, train_y)

imp = importances(rf, train_X, train_y) # permutation

print(imp)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix

from sklearn.tree.export import export_text



dtree_cols=['WomanOrBoyCount2','WomanOrChild','Pclass_Poor','Pclass_Rich']

dtree = DecisionTreeClassifier(max_depth=3).fit(train_X[dtree_cols], train_y)

print(cohen_kappa_score(train_y, dtree.predict(train_X[dtree_cols])))

print(confusion_matrix(train_y, dtree.predict(train_X[dtree_cols])))

tree_rules = export_text(dtree, feature_names=dtree_cols, show_weights=True)

print(tree_rules)
dtree2_cols=['Ticket_Number','WomanOrBoyCount2','WomanOrChild']

dtree2 = DecisionTreeClassifier(max_depth=10).fit(train_X[dtree2_cols], train_y)

print(cohen_kappa_score(train_y, dtree2.predict(train_X[dtree2_cols])))

print(confusion_matrix(train_y, dtree2.predict(train_X[dtree2_cols])))

tree_rules = export_text(dtree2, feature_names=dtree2_cols, show_weights=True)

print(tree_rules)
dtree3_cols=['LastName','WomanOrBoyCount2','WomanOrChild']

dtree3 = DecisionTreeClassifier(max_depth=10).fit(train_X[dtree3_cols], train_y)

print(cohen_kappa_score(train_y, dtree3.predict(train_X[dtree3_cols])))

print(confusion_matrix(train_y, dtree3.predict(train_X[dtree3_cols])))

tree_rules = export_text(dtree3, feature_names=dtree3_cols, show_weights=True)

print(tree_rules)
submission['Survived_dtree'] = dtree.predict(test_onehot[dtree_cols]).astype(int)

submission['Survived_dtree2'] = dtree2.predict(test_onehot[dtree2_cols]).astype(int)

submission['Survived_dtree3'] = dtree3.predict(test_onehot[dtree3_cols]).astype(int)

submission['Survived_gender'] = submission['Survived']

submission['Average_Survived'] = (submission['Survived_dtree'] + submission['Survived_dtree2'] + submission['Survived_dtree3'] + submission['Survived_gender'])/4.0

submission['Survived'] = (submission['Average_Survived'] >= 0.75) * 1
submission = submission[['PassengerId','Survived']]
submission.head()
submission.to_csv('submission.csv', index=False)