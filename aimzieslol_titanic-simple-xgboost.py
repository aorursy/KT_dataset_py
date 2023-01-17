import pandas as p

import numpy as np

import keras as k

import xgboost as xgb



from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder, StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import accuracy_score



from sklearn.metrics import accuracy_score, log_loss
usable_test_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket']

usable_cols = [x for x in usable_test_cols]

usable_cols.extend(['Survived'])
train_df = p.read_csv('datasets/titanic/train.csv', usecols=usable_cols)

test_df  = p.read_csv('datasets/titanic/test.csv', usecols=usable_test_cols)
cols_to_encode_and_dump = ['Sex']



def one_hot_encode(df, cols):

    dummies = []

    for c in cols:

        dummies.append(p.get_dummies(df[c], c))

    return dummies



train_df = train_df.join(one_hot_encode(train_df, cols_to_encode_and_dump))

test_df = test_df.join(one_hot_encode(test_df, cols_to_encode_and_dump))

    

train_df.drop(cols_to_encode_and_dump, axis=1, inplace=True)

test_df.drop(cols_to_encode_and_dump, axis=1, inplace=True)



train_df['Cabin'].fillna('UNKNOWN', inplace=True)

test_df['Cabin'].fillna('UNKNOWN', inplace=True)



train_df['Embarked'].ffill(inplace=True)

test_df['Embarked'].ffill(inplace=True)
cabin_le = LabelEncoder()



unique_cabins = np.append(train_df['Cabin'].unique(), test_df['Cabin'].unique())



cabin_le.fit(unique_cabins)



train_df['Cabin'] = cabin_le.transform(train_df['Cabin'].values)

test_df['Cabin'] = cabin_le.transform(test_df['Cabin'].values)
embarked_le = LabelEncoder()



unique_embarks = np.append(train_df['Embarked'].unique(), test_df['Embarked'].unique())



embarked_le.fit(unique_embarks)



train_df['Embarked'] = embarked_le.transform(train_df['Embarked'].values)

test_df['Embarked'] = embarked_le.transform(test_df['Embarked'].values)
ticket_le = LabelEncoder()



unique_tickets = np.append(train_df['Ticket'].unique(), test_df['Ticket'].unique())



ticket_le.fit(unique_tickets)



train_df['Ticket'] = ticket_le.transform(train_df['Ticket'].values)

test_df['Ticket'] = ticket_le.transform(test_df['Ticket'].values)
scaled_train = p.DataFrame(MinMaxScaler().fit_transform(train_df), columns=train_df.columns)

scaled_test = p.DataFrame(MinMaxScaler().fit_transform(test_df), columns=test_df.columns)
xg_age_booster = xgb.XGBRegressor()



# Grab the training ages and drop the Survived field to match the testing set.



X_entries_with_age = scaled_train[scaled_train['Age'].notna()].copy()

X_entries_with_age.drop(['Survived'], axis=1, inplace=True)



# Grab the test ages.



X_entries_with_age_test = scaled_test[scaled_test['Age'].notna()].copy()



# Merge 'em in a huge list.



X_big_huge_age_list = p.merge(X_entries_with_age, X_entries_with_age_test, how='outer')



# Pop the labels.



y_entries_with_age = X_big_huge_age_list.pop('Age')



# Fit BOTH test/train.



xg_age_booster.fit(X_big_huge_age_list, y_entries_with_age)



# Now grab the blank agest from the training set.



X_entries_wo_age = scaled_train[scaled_train['Age'].isna()].copy()

X_entries_wo_age.drop(['Age', 'Survived'], axis=1, inplace=True)



print(X_entries_wo_age)



training_col_list = X_entries_wo_age.columns



# Predict.



age_results = xg_age_booster.predict(X_entries_wo_age)



# Grab them and merge them back in.



X_entries_wo_age['Age'] = age_results



scaled_train = scaled_train.combine_first(X_entries_wo_age['Age'].to_frame())



# Do the same to the test set.



X_entries_wo_age_test = scaled_test[scaled_test['Age'].isna()].copy()

X_entries_wo_age_test.drop(['Age'], axis=1, inplace=True)



print(training_col_list)



X_entries_wo_age_test = X_entries_wo_age_test[training_col_list]



age_results = xg_age_booster.predict(X_entries_wo_age_test)



X_entries_wo_age_test['Age'] = age_results



scaled_test = scaled_test.combine_first(X_entries_wo_age_test['Age'].to_frame())
skf = StratifiedKFold(n_splits=10)



Y = scaled_train['Survived'].copy()

dropped_scaled_train = scaled_train.drop(['Survived'], axis=1).copy()



xg_surv_booster = xgb.XGBClassifier()



for train_idx, test_idx in skf.split(dropped_scaled_train.values, Y):

    X_train, X_test = dropped_scaled_train.values[train_idx], dropped_scaled_train.values[test_idx]

    y_train, y_test = Y.values[train_idx], Y.values[test_idx]



    xg_surv_booster.fit(X_train, y_train)

    

    train_predictions = xg_surv_booster.predict(X_test)



    acc = accuracy_score(y_test, train_predictions, normalize=True)

    

    print('acc = %.04f%%' % (acc * 100))

    
xg_surv_booster = xgb.XGBClassifier()



Y = scaled_train['Survived'].copy()

dropped_scaled_train = scaled_train.drop(['Survived'], axis=1).copy()



xg_surv_booster.fit(dropped_scaled_train, Y)
reworked_scaled_test = scaled_test[dropped_scaled_train.columns]



results = xg_surv_booster.predict(reworked_scaled_test)
unadulterated_test = p.read_csv('datasets/titanic/test.csv')



scaled_test['Survived'] = results



scaled_test['PassengerId'] = unadulterated_test['PassengerId']

submission_ds = scaled_test[['PassengerId', 'Survived']]

submission_ds.to_csv('submission.csv', index=False)