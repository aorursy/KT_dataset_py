import pandas as pd

import numpy as np
train = pd.read_csv('../input/mobilityanalytics/train_Wc8LBpr.csv')

test = pd.read_csv('../input/mobilityanalytics/test_VsU9xXK.csv')
train.shape, test.shape
train.head()
combine = train.append(test)

combine.shape
combine.isnull().sum()
combine.dtypes
combine.columns
combine['Cancellation_Last_1Month'].value_counts()
bins= [0, 1, 2, 3, 8]

labels = ['None','Once', 'Twice','More_Than_Thrice']

combine['Cancellation_Last_1Month'] = pd.cut(combine['Cancellation_Last_1Month'], bins=bins, labels=labels, right=False)

combine['Cancellation_Last_1Month'].value_counts()
combine['Confidence_Life_Style_Index'].value_counts()
combine['Confidence_Life_Style_Index'].fillna('Unknown', inplace=True)

combine['Confidence_Life_Style_Index'].value_counts()
combine['Customer_Rating'].describe()
combine['Customer_Since_Months'].value_counts()
from sklearn.preprocessing import scale

combine['Customer_Since_Months'].fillna(-1, inplace=True)

combine['Customer_Since_Months'] = scale(combine['Customer_Since_Months'])

combine['Customer_Since_Months'].describe()
combine['Destination_Type'].value_counts()
combine['Gender'].value_counts()
combine['Life_Style_Index'].describe()
combine['Life_Style_Index'].fillna(combine['Life_Style_Index'].mean(), inplace=True)

combine['Life_Style_Index'].describe()
combine['Trip_Distance'].describe()
combine['Trip_Distance'] = np.log(combine['Trip_Distance'])

combine['Trip_Distance'].describe()
combine['Type_of_Cab'].value_counts()
combine['Type_of_Cab'].fillna('Unknown', inplace=True)

combine['Type_of_Cab'].value_counts()
combine['Var1'].describe()
combine['Var1'].fillna(combine['Var1'].mean(), inplace=True)

combine['Var1'] = np.log(combine['Var1'])

combine['Var1'].describe()
combine['Var2'].describe()
combine['Var2'] = np.log(combine['Var2'])

combine['Var2'].describe()
combine['Var3'].describe()
combine['Var3'] = np.log(combine['Var3'])

combine['Var3'].describe()
combine.isnull().sum()
combine = pd.get_dummies(combine.drop('Trip_ID', axis=1))

combine.shape
combine.head()
X = combine[combine['Surge_Pricing_Type'].isnull()!=True].drop(['Surge_Pricing_Type'], axis=1)

y = combine[combine['Surge_Pricing_Type'].isnull()!=True]['Surge_Pricing_Type']



X_test = combine[combine['Surge_Pricing_Type'].isnull()==True].drop(['Surge_Pricing_Type'], axis=1)



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994,

                       objective='multiclass')



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict(x_val)
from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_val, pred_y))

confusion_matrix(y_val,pred_y)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994,

                       objective='multiclass')

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=200)

    pred_y = m.predict(x_val)

    print(i, " err_lgm: ", accuracy_score(y_val, pred_y))

    err.append(accuracy_score(y_val, pred_y))

    pred_test = m.predict(X_test)

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err, 0)
err[3]
submission = pd.DataFrame()

submission['Trip_ID'] = test['Trip_ID']

submission['Surge_Pricing_Type'] = y_pred_tot_lgm[3]

submission.to_csv('LGBM.csv', index=False, header=True)

submission.shape
submission.head()
submission['Surge_Pricing_Type'].value_counts()