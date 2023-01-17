train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Drop the un-necasary columns:

train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

test.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
def check_child(age_gender):

    age, sex = age_gender

    return 'child' if age < 17 else sex



train['Person'] = train[['Age', 'Sex']].apply(check_child, axis=1)

test['Person'] = test[['Age', 'Sex']].apply(check_child, axis=1)



#creating dummies the new person feature for our model

train = pd.concat([train, pd.get_dummies(train.Person, prefix='person')], axis=1)

test = pd.concat([test, pd.get_dummies(test.Person, prefix='person')], axis=1)



train.head()
_, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,12))



#spread

sns.countplot('Person', data=train, ax=ax1)



#survival

sns.countplot('Person', hue='Survived', data=train, ax=ax2)



#mean-survival

person_survival = train[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot('Person', 'Survived', data=person_survival, ax=ax3)



#Drop the original features, they are no more needed.

train.drop(['Sex', 'Person'], inplace=True, axis=1)

test.drop(['Sex', 'Person'], inplace=True, axis=1)