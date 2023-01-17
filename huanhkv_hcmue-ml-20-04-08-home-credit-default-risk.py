import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





from sklearn.feature_selection import SelectPercentile

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from sklearn.ensemble import RandomForestClassifier



import scipy.stats



import os

input_path = '../input/home-credit-default-risk/'

os.listdir(input_path)
app_train = pd.read_csv(input_path + 'application_train.csv')

app_test = pd.read_csv(input_path + 'application_test.csv')

sample_submission = pd.read_csv(input_path + 'sample_submission.csv')



print('Shape app_train:', app_train.shape)

print('Shape app_test:', app_test.shape)



SK_ID_CURR = app_test.iloc[:, 0]

print(app_test.shape, SK_ID_CURR.shape)



app_train.head()
app_test.head()
# bureau_df = pd.read_csv(input_path + 'bureau.csv')

# bureau_balance_df = pd.read_csv(input_path + 'bureau_balance.csv')



# pos_cash_df = pd.read_csv(input_path + 'POS_CASH_balance.csv')

# credit_card_df = pd.read_csv(input_path + 'credit_card_balance.csv')



# prev_app_df = pd.read_csv(input_path + 'previous_application.csv')



# HomeCredit_columns_description_df = pd.read_csv(input_path + 'HomeCredit_columns_description.csv')

# installments_payments_df = pd.read_csv(input_path + 'installments_payments.csv')
app_train['TARGET'].hist()

app_train['TARGET'].value_counts()
# Select columns have dtype object

objct_cols = app_train.select_dtypes(include=object)

objct_cols_list = app_train.select_dtypes(include=object).columns



# Fill value

app_train[objct_cols_list] = app_train[objct_cols_list].fillna('Missing_Data') 

app_test[objct_cols_list] = app_test[objct_cols_list].fillna('Missing_Data') 



# Fill value columns obj by mean of label

for col in objct_cols_list:

    label_mean = app_train.groupby(col).TARGET.mean()

    app_train[col] = app_train[col].map(label_mean).copy()

    app_test[col] = app_test[col].map(label_mean).copy()



# fill nan

app_train = app_train.dropna()

app_test = app_test.fillna(app_train.median())



app_train.shape, app_test.shape
target = app_train.pop('TARGET')



# select 4% columns

select = SelectPercentile(percentile=4)

select.fit(app_train, target)

mask = select.get_support()

df_selected = select.transform(app_train)



masked_list = app_train.columns[np.where(mask==True)]



print(app_train.columns[np.where(mask==True)])

df_selected.shape
target = target.values

X_train, X_test, y_train, y_test = train_test_split(df_selected, target, test_size=0.25, random_state=42)



print('X_train:', X_train.shape)

print('X_test:', X_test.shape)

print('y_train:', y_train.shape)

print('y_test:', y_test.shape)
model_1 = RandomForestClassifier(n_estimators = 10, max_depth= 5, random_state= 42)

model_1.fit(X_train, np.ravel(y_train))



y_train_pred_1 = model_1.predict(X_train)

y_test_pred_1 = model_1.predict(X_test)



print('Accuracy\n', model_1.score(X_test, y_test))

print('\nROC AUC SCORE\n', roc_auc_score(y_test, y_test_pred_1))

print('\nConfusion Matrix\n',confusion_matrix(y_test, y_test_pred_1), '\n\n')



X_train = np.concatenate([X_train, y_train_pred_1.reshape(-1, 1)], 1)

X_test = np.concatenate([X_test, y_test_pred_1.reshape(-1, 1)], 1)



print('Train:', X_train.shape, y_train.shape)

print('Test:', X_test.shape, y_test.shape)
model_2 = RandomForestClassifier(n_estimators = 10, max_depth= 5, random_state= 42)

model_2.fit(X_train, np.ravel(y_train))



y_pred_2 = model_2.predict(X_test)

y_pred_proba_2 = model_2.predict_proba(X_test)



print('Accuracy\n', model_2.score(X_test, y_test))

print('\nROC AUC SCORE\n', roc_auc_score(y_test, y_pred_proba_2[:,1]))

print('\nConfusion Matrix\n',confusion_matrix(y_test, y_pred_2))





fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_2[:,1])



plt.plot(fpr, tpr, marker='o')

plt.xlabel('FPR: False positive rate')

plt.ylabel('TPR: True positive rate')

plt.grid()
app_test = app_test[masked_list]

app_test.shape
test_pred = model_1.predict(app_test)

app_test = np.concatenate([app_test, test_pred.reshape(-1, 1)], 1)
y_pred_proba = model_2.predict_proba(app_test)

y_pred_proba.shape
Submission = pd.DataFrame({'SK_ID_CURR': SK_ID_CURR,'TARGET': y_pred_proba[:,1] })

Submission.to_csv("Submission.csv", index=False)



Submission