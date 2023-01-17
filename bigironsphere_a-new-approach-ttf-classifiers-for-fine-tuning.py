import numpy as np 

import pandas as pd 

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score, mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#I took these from a private kernel of mine but you can add Andrew's Features yourself

train = pd.read_csv('../input/andrews-features/train_X.csv')

test = pd.read_csv('../input/andrews-features/test_X.csv')

y_train = pd.read_csv('../input/andrews-features/train_y.csv').values.flatten()
#remove overlapping segments

def sequential(df, col='time_to_failure', newcol='quake_id', overlap='remove_idx'):

    df[newcol] = np.zeros(len(df))

    df[overlap] = np.zeros(len(df))

    for i in range(1, len(df)):

        if df.loc[i, col] > df.loc[i-1, col]:

            df.loc[i, newcol] = df.loc[i-1, newcol] + 1

            df.loc[i, overlap] = 1

        else:

            df.loc[i, newcol] = df.loc[i-1, newcol]

    return(df)   



train['time_to_failure'] = y_train

train = sequential(train)

print(train.quake_id.describe())

print('Total number of overlapping segments: ', train.remove_idx.sum())
#keep only non-overlapping segments

keep_index = train.loc[train.remove_idx != 1, :].index

train = train.iloc[keep_index].reset_index(drop=True)

y_train = y_train[keep_index]



#save quake ids as numpy array and remove unnecessary columns

quake_ids = train['quake_id'].values

np.save('quake_ids.npy', quake_ids)

np.save('keep_index.npy', keep_index)

train.drop(['remove_idx', 'time_to_failure', 'quake_id'], axis=1, inplace=True)
N_FOLD = 5

SEP = 0.5



folds = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)

feature_importance = np.zeros(len(train.columns))



train_preds = pd.DataFrame()

test_preds = pd.DataFrame()



def ttf_classifier(threshold, X, X_test, df, test_df, y=y_train, sep=SEP, feature_importance=feature_importance):

    

    #y == 1 if TTF lies in specific range

    y = np.logical_and(y >= threshold, y < threshold + sep)

    models = []

    oof = []

    

    for train_index, test_index in folds.split(y):

        

        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y[train_index], y[test_index]

        

        #make and fit simple classifier model

        model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1)

        model.fit(X_train, y_train,

          eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc',

          verbose=0, early_stopping_rounds=500)

        

        models.append(model)

        oof.append(roc_auc_score(y_val, model.predict(X_val)))

        feature_importance += model.feature_importances_/N_FOLD

    

    preds = np.zeros(len(X))

    test_preds = np.zeros(len(X_test))

    

    #predictions for train and test with these models

    #column names lie halfway through the range, i.e. for y==1 at 3s to 4s, column name is '3.5'

    for model in models:

        preds += model.predict(X)

        preds /= len(models)

        df[str(threshold + sep/2)] = preds

        

        test_preds += model.predict(X_test)

        test_preds /= len(models)

        test_df[str(threshold + sep/2)] = test_preds

    

    #return the AUC for the combined classifiers at the target range            

    return(np.asarray(oof).mean())
auc_af = []

thresh = []

#round maximum value down to smallest multiple of 0.5

MAX_Y = np.floor(y_train.max()*2)/2



for i in np.arange(0, MAX_Y, SEP):

    auc_af.append(ttf_classifier(i, X=train, X_test=test, df=train_preds, test_df=test_preds))

    thresh.append(i)   
feature_importance /= len(auc_af)

feat_df = pd.DataFrame({'feature' : train.columns,

                       'importance' : feature_importance}).sort_values('importance', ascending=False)



plt.figure(figsize=(16, 32));

sns.barplot(x='importance', y='feature', data=feat_df.iloc[:, :])

plt.title('Mean Feature Importance')

plt.show()
auc_af_df = pd.DataFrame({'threshold' : np.asarray(thresh) + SEP/2,

                      'auc' : np.asarray(auc_af)})



auc_af_df.plot(kind='line', x = 'threshold', y='auc', figsize=(15, 6))

plt.title('AUC for predicting TTF lies in bin of width 1s')

plt.xlabel('TTF')
#return column name with maximum probability for each row for train and tets

train_preds['Pred'] = train_preds.apply(lambda x: x.argmax(), axis=1)

train_preds['Pred'] = train_preds['Pred'].astype('float16')



test_preds['Pred'] = test_preds.apply(lambda x: x.argmax(), axis=1)

test_preds['Pred'] = test_preds['Pred'].astype('float16')
train_preds['idx'] = train_preds.index

train_preds['ttf'] = y_train

ax = train_preds.plot(kind='scatter',x='idx', y='Pred', figsize=(15, 6), s=1.5, color='b')

train_preds.plot(kind='scatter',x='idx',y='ttf', figsize=(15, 6), s=0.75, color='r', ax=ax)

plt.ylabel('TTF')

plt.title('Predictions/Actual')

plt.show()



print('MAE for classifier: ', mean_absolute_error(y_train, train_preds['Pred'].values))
sub = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')

sub['time_to_failure'] = test_preds['Pred'].values

sub.to_csv('utterly_dreadful_classifier_submission.csv', index=False)
def ttf_ineq_classifier(threshold, X=train, y=y_train):

    

    y = y >= threshold

    models = []

    oof = []

    

    for train_index, test_index in folds.split(y):

        

        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y[train_index], y[test_index]

        

        model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1)

        model.fit(X_train, y_train,

          eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='auc',

          verbose=0, early_stopping_rounds=500)

        models.append(model)

        

        oof.append(roc_auc_score(y_val, model.predict(X_val)))

                           

    return(np.asarray(oof).mean())
SEP=0.1

auc_ineq = []

thresh = []

for i in np.arange(SEP, MAX_Y-1, SEP):

    auc_ineq.append(ttf_ineq_classifier(i))

    thresh.append(i)

    

auc_ineq_df = pd.DataFrame({'threshold' : np.asarray(thresh),

                      'auc' : np.asarray(auc_ineq)})



auc_ineq_df.plot(kind='line', x = 'threshold', y='auc', figsize=(15, 6))

plt.title('AUC for predicting TTF > threshold')