#%%

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


train_file_path = '../input/titanic/train.csv'
test_file_path = '../input/titanic/test.csv'
train = pd.read_csv(train_file_path, index_col='PassengerId')
test = pd.read_csv(test_file_path, index_col='PassengerId')
all = pd.concat([train, test]).reset_index(drop=True)

all.info()

all.head()

#name
names = all.Name
names_df = pd.DataFrame(columns=['Surname', 'Status'])
for name in names:
    surname, lost = name.split(', ')
    lost = lost[:lost.find('(')]
    status, lost_n = lost.split('. ')
    names_df.loc[len(names_df)] = [surname, status]
names_df

names_df.Status[names_df.Status == 'Don'] = 'Mr'
names_df.Status[names_df.Status == 'Dr'] = 'Mr'
names_df.Status[names_df.Status == 'Jonkheer'] = 'Mr'
names_df.Status[names_df.Status == 'Sir'] = 'Mr'
names_df.Status[names_df.Status == 'Major'] = 'Mr'

names_df.Status[names_df.Status == 'Dona'] = 'Mrs'
names_df.Status[names_df.Status == 'Lady'] = 'Mrs'
names_df.Status[names_df.Status == 'Mme'] = 'Mrs'
names_df.Status[names_df.Status == 'the Countess'] = 'Mrs'

names_df.Status[names_df.Status == 'Ms'] = 'Miss'
names_df.Status[names_df.Status == 'Mlle'] = 'Miss'

names_df.Status[names_df.Status == 'Capt'] = 'Other'
names_df.Status[names_df.Status == 'Col'] = 'Other'
names_df.Status[names_df.Status == 'Master'] = 'Other'
names_df.Status[names_df.Status == 'Rev'] = 'Other'
names_df.groupby('Status').count()

all = all.join(names_df).drop('Name', axis=1)
all.head()

all.SibSp.replace([0,], '0', inplace=True)
all.SibSp.replace([1,], '1', inplace=True)
all.SibSp.replace([2,], '2', inplace=True)
all.SibSp.replace([3, 4, 5, 8], 'Many', inplace=True)

all.Parch.replace([0,], '0', inplace=True)
all.Parch.replace([1,], '1', inplace=True)
all.Parch.replace([2,], '2', inplace=True)
all.Parch.replace([3, 4, 5, 6, 9], 'Many', inplace=True)

fig = plt.figure(figsize=(20, 15))
fig.add_subplot(2, 1, 1)
sns.countplot(data=all, x='Parch', hue='Survived')
fig.add_subplot(2, 1, 2)
sns.countplot(data=all, x='Parch')

all.Fare.fillna(all.Fare.median(), inplace=True)

all.groupby(['Sex', 'SibSp', 'Parch']).Age.median()

all.Age = all.groupby(['Sex', 'SibSp', 'Parch']).Age.apply(lambda x: x.fillna(x.median()))

def convert_cabin_data(df):
    pas_id = df.Cabin.index
    cab_dict = {}
    df.Cabin.fillna(value='None', inplace=True)
    for i, cab in enumerate(df.Cabin):
        if cab != 'None':
            if cab.find(' ') != -1:
                cab = cab.split(' ')[1]
            deck = cab[0]
            if cab[1:]:
                cabin_num = int(cab[1:])
            else:
                cabin_num = pd.np.nan
        else:
            deck = 'Unknown'
            cabin_num = pd.np.nan
        cab_dict[pas_id[i]] = [deck, cabin_num]
    cabin = pd.DataFrame.from_dict(cab_dict, orient='index', columns=['Deck', 'Cabin_num'])
    return cabin

cab = convert_cabin_data(all)
all = all.join(cab)
all = all.drop('Cabin', axis=1)
all.head()

all.Deck.replace(['T',], 'A', inplace=True)

sns.set_style('white')
all_survived_grouped_by_deck = all.groupby('Deck').Survived.count()
all_survived = all[all.Survived == 1].groupby('Deck').Survived.count()
survived_percent_grouped_by_deck = all[all.Survived == 1].groupby('Deck').Survived.count()/all_survived_grouped_by_deck
rect = plt.bar(x=survived_percent_grouped_by_deck.index, height=survived_percent_grouped_by_deck)
plt.title('Survivals grouped by Deck')
plt.xlabel('Deck')
plt.ylabel('%')
for i, r in enumerate(rect):
        height = r.get_height()
        r.axes.annotate('{}/{}'.format(all_survived[i], all_survived_grouped_by_deck[i]),
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.show()

all.Deck.replace(['B', 'D', 'E', 'F'], 'B/D/E/F', inplace=True)
all.Deck.replace(['Unknown'], 'G', inplace=True)

all.Embarked.value_counts()
all.Embarked.fillna('S', inplace=True)
all.drop('Cabin_num', inplace=True, axis=1)
all.drop('Surname', inplace=True, axis=1)
all.drop('Ticket', inplace=True, axis=1)

processed_train = all.iloc[:891]
processed_train.set_index(pd.Series([x for x in range(1, 892)], name='PassengerId'), inplace=True)
processed_test = all.iloc[891:].drop('Survived', axis=1)
processed_test.set_index(pd.Series([x for x in range(892, 1310)], name='PassengerId'), inplace=True)

num_features_cols = ['Age', 'Fare']
cat_features_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Status', 'Deck']

plt.figure(figsize=(15, 10))
sns.barplot(x=processed_train['Survived'].value_counts(normalize=True).index, y=processed_train['Survived'].value_counts(normalize=True))
sns.set_style('white')
plt.grid()
plt.title('Distribution of Survivors in a Training Sample', fontdict={'fontsize': 25})
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(ticks=[0, 1], labels=['Not Survived', 'Survived'])
plt.ylabel('Percent of Survived', fontdict={'fontsize': 20})
plt.show()

fig = plt.figure(figsize=(25,35))
for i in range(len(cat_features_cols)):
    fig.add_subplot(4, 2, i+1)
    sns.countplot(x=cat_features_cols[i], hue='Survived', data=processed_train)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(['Not Survived', 'Survived'])

from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

#  OneHotEncoder for categorical data
transformed_list = []
for feature in cat_features_cols:
    new_feature_colls = ['{}_{}'.format(feature, x+1) for x in range(processed_train[feature].nunique()) ]
    transformed_cat_df = pd.DataFrame(OH_encoder.fit_transform(processed_train[feature].values.reshape(-1, 1)), columns=new_feature_colls)
    transformed_cat_df.set_index(pd.Series([x for x in range(1, 892)], name='PassengerId'), inplace=True)
    transformed_list.append(transformed_cat_df)
new_train = processed_train[['Survived', 'Age', 'Fare']]
for feature_df in transformed_list:
    new_train = new_train.join(feature_df)
new_train.tail()

transformed_list = []
for feature in cat_features_cols:
    new_feature_colls = ['{}_{}'.format(feature, x+1) for x in range(processed_test[feature].nunique()) ]
    transformed_cat_df = pd.DataFrame(OH_encoder.fit_transform(processed_test[feature].values.reshape(-1, 1)), columns=new_feature_colls)
    transformed_cat_df.set_index(pd.Series([x for x in range(892, 1310)], name='PassengerId'), inplace=True)
    transformed_list.append(transformed_cat_df)
new_test = processed_test[['Age', 'Fare']]
for feature_df in transformed_list:
    new_test = new_test.join(feature_df)
new_test.tail()

x = new_train.drop('Survived', axis=1)
y = new_train.Survived
x_t = new_test

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

def modelfit(alg, fit_param_name, fit_param_values):
    fit_param_name = fit_param_name.rstrip()
    best_param = None
    best_acc = None
    for param_value in fit_param_values:
        alg.set_params(**{fit_param_name: param_value})
        temp_test_acc = []
        for train_index, test_index in skf.split(x, y):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            alg.fit(X_train, y_train)
            temp_test_acc.append(accuracy_score(y_test, alg.predict(X_test)))
        acc = pd.Series(temp_test_acc).mean()*100
        print("Accuracy : {:4.2f}%, {}: {:<}".format(acc, fit_param_name, param_value))
        if best_acc:
            if acc > best_acc:
                best_acc = acc
                best_param = param_value
        else:
            best_acc = acc
            best_param = param_value
    print(40*'-')
    print('Best Accuracy= {:4.2f}% for {}= {:<}'.format(best_acc, fit_param_name, best_param))

#RandomForest
#baseline model
from sklearn.ensemble import RandomForestClassifier
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)
temp_test_acc = []
for train_index, test_index in skf.split(x, y):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rfc.fit(X_train, y_train)
    temp_test_acc.append(accuracy_score(y_test, rfc.predict(X_test)))
acc = pd.Series(temp_test_acc).mean()*100
print("Accuracy score: {:.2f}%".format(acc))

modelfit(RandomForestClassifier(random_state=42,
                                n_jobs=-1),
         'n_estimators',
         [a for a in range(490, 510, 1)])

modelfit(RandomForestClassifier(random_state=42,
                                n_jobs=-1,
                                n_estimators=497),
         'max_depth',
         [2, 3, 4, 5, 6, 7, 8, 9])

modelfit(RandomForestClassifier(random_state=42,
                                n_jobs=-1,
                                n_estimators=497,
                                max_depth=5),
         'max_features',
         ['auto', 1, 3, 5, 7, 10, 12, 15, 16, 17, 18])

#XGBoost
#baseline
from xgboost import XGBClassifier

xgbc = XGBClassifier(random_state=42)
temp_test_acc = []
for train_index, test_index in skf.split(x, y):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    xgbc.fit(X_train, y_train)
    temp_test_acc.append(accuracy_score(y_test, xgbc.predict(X_test)))
acc = pd.Series(temp_test_acc).mean()*100
print("Accuracy score: {:.2f}%".format(acc))

modelfit(XGBClassifier(random_state=42, n_jobs=-1),
         'n_estimators',
         [a for a in range(7, 16, 1)])

modelfit(XGBClassifier(n_estimators=11, random_state=42, n_jobs=-1),
         'learning_rate',
         [a/100 for a in range(1, 36, 1)])

modelfit(XGBClassifier(learning_rate=0.23,
                       n_estimators=11,
                       random_state=42,
                       n_jobs=-1),
         'max_depth',
         [2, 3, 4, 5, 6, 7, 8, 9, 15])

modelfit(XGBClassifier(learning_rate=0.23,
                       n_estimators=11,
                       random_state=42,
                       n_jobs=-1,
                       max_depth=8),
         'gamma',
         [a/100 for a in range(0, 36, 1)])
xgbc = XGBClassifier(learning_rate=0.23,
                       n_estimators=11,
                       random_state=42,
                       n_jobs=-1,
                       max_depth=8,
                       gamma=0.09)
xgbc.fit(x, y)

imp = pd.DataFrame({'Feature_Importance': xgbc.feature_importances_}, index=x.columns).sort_values(by='Feature_Importance', ascending=False)
imp.plot(kind='bar')
pred = xgbc.predict(x_t)
submission = pd.DataFrame({'PassengerId':x_t.index, 'Survived': pred.astype('int')})
submission.to_csv('Submissions.csv', header=True, index=False)