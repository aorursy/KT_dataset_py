import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_train.info()
for c in df_train.select_dtypes('object').columns:

    print('='*75)

    print('column "{}" has {} unique values and {} missing values'.format(c,len(df_train[c].drop_duplicates()),

                                                          df_train[c].isnull().sum()))

    print(df_train[c].value_counts().head(5))
df_train['Pclass'].value_counts()
def model(df):

    # get the columns for independent variables (X) and target (y)

    X = df.drop(['PassengerId','Survived'],axis=1).values

    y = df['Survived'].values



    # this variable store the column names that will be used later to plot the feature importance

    fcol = df.drop(['PassengerId','Survived'],axis=1).columns



    # Create Stratified 3 fold split, make sure the random seed is set to make the result reproducible

    skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=666)



    # This is parameters for LGBM, use binary error as a metric and set the random seed

    param = {

             'objective':'binary',

             "metric": 'binary_error',

             "random_state": 4590

    }



    # These variables is used to store the scores and feature importance

    evals = {}

    avg_tr_score = 0

    avg_val_score = 0

    df_feats = pd.DataFrame()



    for fold, (trn_idx, val_idx) in enumerate(skf.split(X,y)):

        print('='*75)



        # load the data as lgb dataset

        trn_data = lgb.Dataset(X[trn_idx], label=y[trn_idx])

        val_data = lgb.Dataset(X[val_idx], label=y[val_idx])



        # train the model, I set early_stopping_rounds = 500 to make sure we get the optimal score, it's probably an overkill though

        num_round = 10000

        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 

                        verbose_eval=100, early_stopping_rounds = 500,evals_result = evals)



        # add the scores for this fold to the final score

        avg_tr_score += evals['training']['binary_error'][clf.best_iteration]/skf.n_splits

        avg_val_score += evals['valid_1']['binary_error'][clf.best_iteration]/skf.n_splits



        # add the feature importance data for this fold to the final feature importance dataframe

        # Note: I get both the feature importance by 'split' and 'information gain'

        df_feats_fold = pd.DataFrame()

        df_feats_fold['feature'] = fcol

        df_feats_fold['importance'] = clf.feature_importance(importance_type='split')

        df_feats_fold['type'] = 'split'

        df_feats_fold['fold'] = fold + 1

        df_feats = pd.concat([df_feats_fold, df_feats], axis=0)



        df_feats_fold = pd.DataFrame()

        df_feats_fold['feature'] = fcol

        df_feats_fold['importance'] = clf.feature_importance(importance_type='gain')

        df_feats_fold['type'] = 'gain'

        df_feats_fold['fold'] = fold + 1

        df_feats = pd.concat([df_feats_fold, df_feats], axis=0)



    print('='*75)    

    print('Average train score: {:.3f} Average validation score: {:.3f}'.format(avg_tr_score,avg_val_score))

    

    # Plot the feature importances

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,10))

    sns.barplot(x='importance',y='feature',

                data=df_feats[df_feats['type']=='split'].sort_values('importance',ascending=False),ax=ax1)

    sns.barplot(x='importance',y='feature',

                data=df_feats[df_feats['type']=='gain'].sort_values('importance',ascending=False),ax=ax2)



    ax1.set_title('Feature importance by split')

    ax2.set_title('Feature importance by gain')



    plt.tight_layout()

    plt.show()
df_train_clean = df_train[['PassengerId','Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']]

df_train_clean.head()
df_train_clean['Sex'] = np.where(df_train_clean['Sex']=='male',1,0).astype('int64')



m = {'S':1,'C':2,'Q':3}

df_train_clean['Embarked'] = df_train_clean['Embarked'].map(m)

df_train_clean['Embarked'] = df_train_clean['Embarked'].fillna(1).astype('int64')

df_train_clean.head()
clf = model(df_train_clean)
sns.distplot(df_train_clean['Fare'])
plt.figure(figsize=(15,5))

sns.countplot(pd.cut(df_train['Fare'],100,labels=False))
df_train_clean = df_train[['PassengerId','Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked']]

df_train_clean.head()
df_train_clean['Sex'] = np.where(df_train_clean['Sex']=='male',1,0).astype('int64')



m = {'S':1,'C':2,'Q':3}

df_train_clean['Embarked'] = df_train_clean['Embarked'].map(m)

df_train_clean['Embarked'] = df_train_clean['Embarked'].fillna(1).astype('int64')

df_train_clean.head()
df_train_clean['Fare'] = pd.cut(df_train_clean['Fare'],100,labels=False)

df_train_clean['Fare'] = np.where(df_train_clean['Fare']>20,21,df_train_clean['Fare'])

df_train_clean.head()
plt.figure(figsize=(15,5))

sns.countplot(df_train_clean['Fare'])
model(df_train_clean)
df_train_clean = df_train[['PassengerId','Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']]

df_train_clean.head()
df_train_clean['Sex'] = np.where(df_train_clean['Sex']=='male',1,0).astype('int64')



m = {'S':1,'C':2,'Q':3}

df_train_clean['Embarked'] = df_train_clean['Embarked'].map(m)

df_train_clean['Embarked'] = df_train_clean['Embarked'].fillna(1).astype('int64')

df_train_clean['Age'] = df_train_clean['Age'].fillna(-1)

df_train_clean.head()
model(df_train_clean)
sns.distplot(df_train_clean['Age'])
df_train_clean = df_train[['PassengerId','Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']]

df_train_clean.head()
df_train_clean['Sex'] = np.where(df_train_clean['Sex']=='male',1,0).astype('int64')

df_train_clean['Family'] = df_train_clean['Parch'] + df_train_clean['SibSp']



m = {'S':1,'C':2,'Q':3}

df_train_clean['Embarked'] = df_train_clean['Embarked'].map(m)

df_train_clean['Embarked'] = df_train_clean['Embarked'].fillna(1).astype('int64')

df_train_clean['Age'] = df_train_clean['Age'].fillna(-1)

df_train_clean.head()
model(df_train_clean)