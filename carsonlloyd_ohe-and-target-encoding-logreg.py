import pandas as pd

import numpy as np

import category_encoders as ce



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, KFold

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, KBinsDiscretizer, FunctionTransformer

from sklearn.model_selection import cross_val_score



import warnings

warnings.filterwarnings("ignore")



import gc; gc.enable()
df = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test_df = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
df.head()
df.describe()
df.isna().sum()
#filling missing values

for col in df:

    if df[col].isna().sum() > 0:

        df[col] = df[col].fillna(df[col].mode()[0])
df.isna().sum()
#One Hot Encoding
y = df['target']

X_train = df.drop(['target', 'id'], axis=1)

X_test = test_df.drop(['id'], axis=1)
combined_data = pd.concat([X_train, X_test], axis=0, sort=False)

combined_data = pd.get_dummies(combined_data, columns=combined_data.columns, drop_first=True, sparse=True)

X_train = combined_data.iloc[: len(df)]

X_test = combined_data.iloc[len(df): ]
print(f'Shape of training dataset: {X_train.shape}')

print(f'Shape of test dataset: {X_test.shape}')
#compress data to make everything run faster

X_train = X_train.sparse.to_coo().tocsr()

X_test = X_test.sparse.to_coo().tocsr()
x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.30)
lr = LogisticRegression(verbose = 100, max_iter = 600, C=0.5, solver='lbfgs')



lr.fit(x_train, y_train)

train_preds = lr.predict_proba(x_train)[:,1]

test_preds = lr.predict_proba(x_test)[:,1]



print("AUC:"); print("="*len("AUC:"))

print("TRAIN:", roc_auc_score(y_train, train_preds))

print("TEST:", roc_auc_score(y_test, test_preds))
#Target Encoding
df = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test_df = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
#fill missing values

for col in df:

    if df[col].isna().sum() > 0:

        df[col] = df[col].fillna(df[col].mode()[0])
used_cols = [c for c in df.columns.tolist() if c not in ['target', 'id']]
#create a target encoder

ce_target_encoder = ce.TargetEncoder(cols = used_cols, smoothing=.3)
target = df['target']

te_df = df.drop(['target', 'id'], axis=1)

te_test_df = test_df.drop('id', axis=1)
ce_target_encoder.fit(te_df, target)
te_df = ce_target_encoder.transform(te_df)

te_test_df = ce_target_encoder.transform(te_test_df)
y = target

X_train = te_df

X_test = te_test_df
test_id = test_df['id']
x_train, x_test, y_train, y_test = train_test_split(X_train, y, test_size=0.30)
lr = LogisticRegression(verbose = 100, max_iter = 600, C=0.5, solver='lbfgs')



lr.fit(x_train, y_train)

train_preds = lr.predict_proba(x_train)[:,1]

test_preds = lr.predict_proba(x_test)[:,1]



print("AUC:"); print("="*len("AUC:"))

print("TRAIN:", roc_auc_score(y_train, train_preds))

print("TEST:", roc_auc_score(y_test, test_preds))
submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

submission["id"] = test_id

submission["target"] = lr.predict_proba(X_test)[:, 1]

submission.to_csv("submission.csv", index=False)