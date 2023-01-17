import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')

df_subm=pd.read_csv('../input/sample_submission.csv')
df_train.head()
df_train.shape, df_test.shape, df_subm.shape
df_train.label.value_counts(normalize=True)
X = df_train.drop(columns=['label'])

y = df_train.label
# # CV training with multiple Algos

# models = [

#           LogisticRegression(n_jobs=-1, random_state=6), 

#           XGBClassifier(random_state=5, n_jobs=-1),

#           ExtraTreesClassifier(random_state=97, n_estimators=100, n_jobs=-1),

#           LGBMClassifier(objective='multiclass', random_state=5, n_jobs=-1)

#          ]

# CV = 5

# cv_df = pd.DataFrame(index=range(CV * len(models)))

# entries = []

# for model in models:

#     model_name = model.__class__.__name__

#     print('{} model Training started.'.format(model_name))

#     acc = cross_val_score(model, X, y, scoring='accuracy', cv=CV)

#     for cv_idx, a in enumerate(acc):

#         entries.append((model_name, cv_idx, a))

#     print('{} model Training Done.'.format(model_name))



# cv_df = pd.DataFrame(entries, columns=['model_name', 'cv_idx', 'Accuracy'])

# print('Cross-Validation complete!')
# cv_df.groupby(['model_name'])['Accuracy'].mean()
model = LGBMClassifier(objective='multiclass', random_state=5, n_jobs=-1)
# StratifiedKFold and Model Training & Evaluation

kfold = 5

skf = StratifiedKFold(n_splits=kfold,shuffle=True,random_state=7)

scores_df = pd.DataFrame(index=range(kfold))

df_row = []

model_name = model.__class__.__name__

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    X_train, X_val = X.loc[train_idx], X.loc[test_idx]

    y_train, y_val = y.loc[train_idx], y.loc[test_idx]

    print('[Fold: {}/{}]'.format(i + 1, kfold))

    

    model.fit(X_train,y_train)

    pred_y = model.predict(X_val)

    train_acc = model.score(X_train, y_train)

    test_acc = accuracy_score(y_val, pred_y)

       

    df_row.append((model_name, i, test_acc, train_acc))

        

print('Training Done!')
scores_df = pd.DataFrame(df_row, columns=['model_name', 'fold_idx', 'Test_acc', 'Train_acc'])

scores_df.sort_values(by=['model_name', 'fold_idx'], inplace=True)

scores_df.reset_index(drop=True, inplace=True)
scores_df
scores_df.groupby(['model_name'])['Test_acc'].mean()
test_pred = model.predict(df_test)

df_subm['Label'] = test_pred

df_subm.to_csv('submission_LGM.csv', index=False)

print(os.listdir("../working"))
df_subm.head()