# Load libraries

import numpy as np

import pandas as pd



import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')
# Import data

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



df_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)

df_s = [df_train, df_test]
display(df_train.head())

print(df_train.info())

display(df_train.describe()) # numerical values

display(df_train.describe(include=['O'])) # object columns
print(df_test.info())

display(df_test.head())
display(df_train[['Age', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Survived']].corrwith(df_train['Age']))



age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']

print(age_by_pclass_sex)

print('Median age of all passengers: {}'.format(df_all['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass groups

df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all['Embarked'] = df_all['Embarked'].fillna('S')
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

df_all['Fare'] = df_all['Fare'].fillna(med_fare)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_all.loc[df_all.query('Deck == "T"').index, 'Deck'] = 'A'



df_all['Deck'].value_counts()
df_all.drop(['Cabin'], inplace=True, axis=1)

df_train, df_test = df_all.loc[:890], df_all.loc[891:].drop(['Survived'], axis=1)
fig, axs = plt.subplots(2, figsize=(15, 15))



sns.heatmap(df_train.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

sns.heatmap(df_test.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

    

axs[0].set_title('Training Set Correlations', size=15)

axs[1].set_title('Test Set Correlations', size=15)
cont_features = ['Age', 'Fare']

surv = df_train['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(cont_features):    

    # Distribution of survival in feature

    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])

    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    

    # Distribution of feature in dataset

    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

    

    for j in range(2):        

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

    

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

        

plt.show()
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=df_train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
df_all['Fare'] = pd.qcut(df_all['Fare'], 10)
df_all['Age'] = pd.qcut(df_all['Age'], 10)
# Plot to see each group. There is also an unusual group.

fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Age', hue='Survived', data=df_all)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)



plt.show()
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1



# Plot to see each group

fig, axs = plt.subplots(figsize=(12, 10), ncols=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=df_all['FamilySize'].value_counts().index, y=df_all['FamilySize'].value_counts().values, ax=axs[0])

sns.countplot(x='FamilySize', hue='Survived', data=df_all, ax=axs[1])



axs[0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[1].set_title('Survival Counts in Family Size ', size=20, y=1.05)
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df_all['Is_Married'] = 0

df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1



# Group to lower cardinality

df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



fig, axs = plt.subplots(figsize=(15, 10))

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs)

axs.set_title('Title Feature Value Counts After Grouping', size=15, y=1.05)



plt.show()
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
df_all = df_all.drop(['Name', 'Ticket'], axis=1)

df_train, df_test = df_all.loc[:890], df_all.loc[891:]

dfs =[df_train, df_test]
ordinal_features = ['Age', 'Fare', 'Sex']



for df in dfs:

    for feature in ordinal_features:        

        df[feature] = LabelEncoder().fit_transform(df[feature])
onehot_features = ['Embarked', 'Deck', 'Title']



df_train = (df_train.join(pd.concat([pd.get_dummies(df_train[feature], prefix=feature) for feature in onehot_features], axis=1))

          .drop(onehot_features, axis=1))



df_test = (df_test.join(pd.concat([pd.get_dummies(df_test[feature], prefix=feature) for feature in onehot_features], axis=1))

          .drop(onehot_features, axis=1))
roc, oob, feature_importance_df = {}, {}, pd.DataFrame()



TRAIN, LABEL = df_train.drop(['Survived', 'PassengerId'], axis=1), df_train['Survived']

TEST = df_test.drop(['PassengerId', 'Survived'], axis=1)

preds, oof_preds = np.zeros(TRAIN.shape[0]), np.zeros(TEST.shape[0])



cv = StratifiedKFold(n_splits=5, shuffle=True)

for i, (train_idx, val_idx) in enumerate(cv.split(TRAIN, LABEL)):

    X_train, y_train = TRAIN.iloc[train_idx], LABEL.iloc[train_idx]

    X_val, y_val = TRAIN.iloc[val_idx], LABEL.iloc[val_idx]



    gbm = RandomForestClassifier(criterion='gini', n_estimators=1750, max_depth=7, min_samples_split=6, min_samples_leaf=6, max_features='auto',

                                 oob_score=True,

                                 random_state=42,

                                 n_jobs=-1,

                                 verbose=False).fit(X_train, y_train)



    y_pred = gbm.predict(X_val)

    y_pred_proba = gbm.predict_proba(X_val)[:, 1]

        

    preds[val_idx] = y_pred_proba

    oof_preds += gbm.predict_proba(TEST)[:, 1] / cv.n_splits # average probability of output

    

    roc['FOLD_' + str(i+1)] = roc_auc_score(y_val, y_pred_proba)

    oob['FOLD_' + str(i+1)] = gbm.oob_score_

    

    # For create feature importances

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = TRAIN.columns

    fold_importance_df["importance"] = gbm.feature_importances_

    fold_importance_df["fold"] = i + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_val, y_pred_proba)))

    print('Fold %2d OOB : %.6f' % (i + 1, gbm.oob_score_), '\n')

    

# Resulting

roc_auc = roc_auc_score(LABEL, preds)

print('Avg AUC score:', roc_auc)

print('Out-of-bag score:', np.mean(list(oob.values())))
def display_importances(feature_importance_df_):



    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(

        by="importance", ascending=False)[:20].index

    

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    

    plt.figure(figsize=(12,8))

    sns.barplot(x="importance", y="feature", 

                data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()



display_importances(feature_importance_df_=feature_importance_df)
model = RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1,

                                           verbose=False) .fit(TRAIN, LABEL)



print(model.oob_score_)

output = model.predict(TEST).astype(int)
submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': (oof_preds >= 0.5).astype(int)}) # or output as single model

time = pd.Timestamp.now().strftime("%Y%m%d_%H_%M_%S")

submission.to_csv(time + '.csv', index=False)