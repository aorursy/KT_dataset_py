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
display(df_train.head())

print(df_train.info())

display(df_train.describe()) # numerical values

display(df_train.describe(include=['O'])) # object columns
print(df_test.info())

display(df_test.head())
to_drop = ['Name', 'Ticket']

for df in [df_train, df_test]:

    df.drop(to_drop, axis=1, inplace=True)
# Sex

sex_map = {'male': 1, 'female': 0}

for df in [df_train, df_test]:

    df['Sex'] = df['Sex'].map(sex_map)



    # Preprocess Cabin

    df['Cabin'] = df['Cabin'].str[0]

    

# Dummies dataframe with Cabin and Embarked

df_train = (df_train

            .join(pd.get_dummies(df_train.Cabin, dummy_na=True, prefix='Cabin'))

            .join(pd.get_dummies(df_train.Embarked, prefix='Embarked'))

            .drop(['Cabin', 'Embarked'], axis=1))



df_test = (df_test

            .join(pd.get_dummies(df_test.Cabin, dummy_na=True, prefix='Cabin'))

            .join(pd.get_dummies(df_test.Embarked, prefix='Embarked'))

            .drop(['Cabin', 'Embarked'], axis=1))
selected_columns = np.intersect1d(df_train.columns, df_test.columns)
acc, roc, feature_importance_df = {}, {}, pd.DataFrame()



TRAIN, LABEL = df_train[selected_columns].drop('PassengerId', axis=1), df_train['Survived']

TEST = df_test[selected_columns].drop(['PassengerId'], axis=1)

preds, oof_preds = np.zeros(TRAIN.shape[0]), np.zeros(TEST.shape[0])



cv = StratifiedKFold(n_splits=5, shuffle=True)

for i, (train_idx, val_idx) in enumerate(cv.split(TRAIN, LABEL)):

    X_train, y_train = TRAIN.iloc[train_idx], LABEL.iloc[train_idx]

    X_val, y_val = TRAIN.iloc[val_idx], LABEL.iloc[val_idx]



    gbm = lgb.LGBMClassifier(random_state=42, learning_rate=0.01, n_estimators=10000).fit(X_train, y_train, 

                                                                                          eval_set=[(X_train, y_train), (X_val, y_val)],

                                                                                          eval_metric='auc',

                                                                                          early_stopping_rounds=1000,

                                                                                          verbose=200)



    y_pred = gbm.predict(X_val)

    y_pred_proba = gbm.predict_proba(X_val)[:, 1]

        

    preds[val_idx] = y_pred_proba

    oof_preds += gbm.predict_proba(TEST)[:, 1] / cv.n_splits # average probability of output

    

    acc['FOLD_' + str(i+1)] = accuracy_score(y_val, y_pred)

    roc['FOLD_' + str(i+1)] = roc_auc_score(y_val, y_pred_proba)

    

    # For create feature importances

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = TRAIN.columns

    fold_importance_df["importance"] = gbm.feature_importances_

    fold_importance_df["fold"] = i + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_val, y_pred_proba)))

    print('Fold %2d ACC : %.6f' % (i + 1, accuracy_score(y_val, y_pred)))

    

# Resulting

roc_auc = roc_auc_score(LABEL, preds)

print('Avg AUC score:', roc_auc)
def display_importances(feature_importance_df_):



    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(

        by="importance", ascending=False)[:15].index

    

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    

    plt.figure(figsize=(12, 8))

    sns.barplot(x="importance", y="feature", 

                data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()



display_importances(feature_importance_df_=feature_importance_df)
submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': (oof_preds >= 0.5).astype(int)}) # or output as single model

time = pd.Timestamp.now().strftime("%Y%m%d_%H_%M_%S")

submission.to_csv(time + '.csv', index=False)