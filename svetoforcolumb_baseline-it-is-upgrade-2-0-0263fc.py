# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import confusion_matrix, classification_report, f1_score



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
os.listdir('/kaggle/input')
# !python3 /kaggle/input/itis-hackathon/make_dataset_kaggle.py /kaggle/input/itis-hackathon/
interim_test = pd.read_csv('/kaggle/working/interim_test.csv')

interim_train_09 = pd.read_csv('/kaggle/working/interim_train_09.csv')

interim_train_10 = pd.read_csv('/kaggle/working/interim_train_10.csv')
interim_train_09 = interim_train_09.drop(columns=['ID','SERVICE_INT_ID',

                                                  'ADMIN_QUESTION_INT_ID',

                                                  'FEATURE_INT_ID','CHANNEL_INT_ID',

                                                  'ACTIVATE_DATE', 'PHYZ_TYPE',

                                                  'CITY_NAME'])
interim_train_10 = interim_train_10.drop(columns=['ID','SERVICE_INT_ID',

                                                  'ADMIN_QUESTION_INT_ID',

                                                  'FEATURE_INT_ID','CHANNEL_INT_ID',

                                                  'ACTIVATE_DATE', 'PHYZ_TYPE',

                                                  'CITY_NAME'])
interim_test = interim_test.drop(columns=['ACTIVATE_DATE', 'PHYZ_TYPE',

                                                  'CITY_NAME'])
interim_train_09.fillna(0, inplace=True)

interim_train_10.fillna(0, inplace=True)

interim_test.fillna(0, inplace=True)
def align_data(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    """Согласование признаков у train и test датасетов

    

    Arguments:

        train {pd.DataFrame} -- train датасет

        test {pd.DataFrame} -- test датасет

    Returns:

        train {pd.DataFrame}, test {pd.DataFrame} - датасеты с одинаковыми признаками

    """

    intersect_list = np.intersect1d(train.columns, test.columns)

    if "TARGET" not in intersect_list:

        train = train[np.append(intersect_list, "TARGET")]

    else:

        train = train[intersect_list]

    test = test[intersect_list]

    return train, test
train_09, test_10 = align_data(interim_train_09, interim_train_10)
train_09.shape, test_10.shape
fin_incidents_train_09 =  pd.read_csv('/kaggle/input/itis-hackathon/train_09_raw_feat/fin_incidents_train_09.csv', parse_dates=['REG_DATE'])

fin_incidents_train_09 = fin_incidents_train_09.drop('ID',axis = 1)

fin_incidents_train_09['REG_DATE'] = fin_incidents_train_09['REG_DATE'].dt.day



# payments_train_10['expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day < 15 else 0)

# payments_train_10['not_expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day > 15 else 0)

# fin_incidents_train_09 = fin_incidents_train_09.drop('NEW_PAY_DATE',axis = 1)



one_hot = pd.get_dummies(fin_incidents_train_09['SERVICE_INT_ID'], 

                         prefix='SERVICE_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('SERVICE_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)





one_hot = pd.get_dummies(fin_incidents_train_09['ADMIN_QUESTION_INT_ID'], 

                         prefix='ADMIN_QUESTION_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('ADMIN_QUESTION_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



one_hot = pd.get_dummies(fin_incidents_train_09['FEATURE_INT_ID'], 

                         prefix='FEATURE_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('FEATURE_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



# one_hot = pd.get_dummies(fin_incidents_train_09['REG_DATE'], 

#                          prefix='REG_DATE')

# fin_incidents_train_09 = fin_incidents_train_09.drop('REG_DATE',axis = 1)

# fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



for column in fin_incidents_train_09.columns:

#     print(column)

    train_09[column] = np.nan

    

train_09 = train_09.join(fin_incidents_train_09, on='USER_ID', rsuffix='_other')

train_09.fillna(0, inplace=True)
# fin_incidents_train_10['REG_DATE'].dt.day# month + ' ' + fin_incidents_train_10['REG_DATE'].dt.day
fin_incidents_train_10 =  pd.read_csv('/kaggle/input/itis-hackathon/train_10_raw_feat/fin_incidents_train_10.csv', parse_dates=['REG_DATE'])

fin_incidents_train_10 = fin_incidents_train_10.drop('ID',axis = 1)

fin_incidents_train_10['REG_DATE'] = fin_incidents_train_10['REG_DATE'].dt.day



# payments_train_10['expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day < 15 else 0)

# payments_train_10['not_expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day > 15 else 0)

# fin_incidents_train_10 = fin_incidents_train_10.drop('NEW_PAY_DATE',axis = 1)



one_hot = pd.get_dummies(fin_incidents_train_10['SERVICE_INT_ID'], 

                         prefix='SERVICE_INT_ID')

fin_incidents_train_10 = fin_incidents_train_10.drop('SERVICE_INT_ID',axis = 1)

fin_incidents_train_10 = fin_incidents_train_10.join(one_hot)





one_hot = pd.get_dummies(fin_incidents_train_10['ADMIN_QUESTION_INT_ID'], 

                         prefix='ADMIN_QUESTION_INT_ID')

fin_incidents_train_10 = fin_incidents_train_10.drop('ADMIN_QUESTION_INT_ID',axis = 1)

fin_incidents_train_10 = fin_incidents_train_10.join(one_hot)



one_hot = pd.get_dummies(fin_incidents_train_10['FEATURE_INT_ID'], 

                         prefix='FEATURE_INT_ID')

fin_incidents_train_10 = fin_incidents_train_10.drop('FEATURE_INT_ID',axis = 1)

fin_incidents_train_10 = fin_incidents_train_10.join(one_hot)





for column in fin_incidents_train_10.columns:

#     print(column)

    test_10[column] = np.nan

    

test_10 = test_10.join(fin_incidents_train_10, on='USER_ID', rsuffix='_other')

test_10.fillna(0, inplace=True)
train_09
test_10
def fit_and_pred_logreg(train, test):

    """Fit and predict LogisticRegression

    

    Arguments:

        train {pd.DataFrame} -- processed train dataset

        test {pd.DataFrame} -- processed test dataset

    

    Returns:

        model {sklearn.BaseEstimator} -- fit sklearn model

        y_pred {np.array} -- predictions

    """

    model = LogisticRegression(class_weight="balanced", random_state=17, n_jobs=-1)

    x_train = train.drop(columns=["TARGET"])

    y_train = train.TARGET

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(test)



    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)



    return model, y_pred
y_test = test_10.TARGET

a, b = align_data(train_09, test_10)

model, y_pred = fit_and_pred_logreg(a, b.drop(columns='TARGET'))
print(classification_report(y_true=y_test, y_pred=y_pred))

print(confusion_matrix(y_true=y_test, y_pred=y_pred))
train_10, test = align_data(interim_train_10, interim_test)
fin_incidents_train_09 =  pd.read_csv('/kaggle/input/itis-hackathon/train_09_raw_feat/fin_incidents_train_09.csv', parse_dates=['REG_DATE'])

fin_incidents_train_09 = fin_incidents_train_09.drop('ID',axis = 1)

fin_incidents_train_09['REG_DATE'] = fin_incidents_train_09['REG_DATE'].dt.day



# payments_train_10['expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day < 15 else 0)

# payments_train_10['not_expired'] = payments_train_10['NEW_PAY_DATE'].apply(lambda x: x if x.day > 15 else 0)

# fin_incidents_train_09 = fin_incidents_train_09.drop('NEW_PAY_DATE',axis = 1)



one_hot = pd.get_dummies(fin_incidents_train_09['SERVICE_INT_ID'], 

                         prefix='SERVICE_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('SERVICE_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)





one_hot = pd.get_dummies(fin_incidents_train_09['ADMIN_QUESTION_INT_ID'], 

                         prefix='ADMIN_QUESTION_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('ADMIN_QUESTION_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



one_hot = pd.get_dummies(fin_incidents_train_09['FEATURE_INT_ID'], 

                         prefix='FEATURE_INT_ID')

fin_incidents_train_09 = fin_incidents_train_09.drop('FEATURE_INT_ID',axis = 1)

fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



# one_hot = pd.get_dummies(fin_incidents_train_09['REG_DATE'], 

#                          prefix='REG_DATE')

# fin_incidents_train_09 = fin_incidents_train_09.drop('REG_DATE',axis = 1)

# fin_incidents_train_09 = fin_incidents_train_09.join(one_hot)



for column in fin_incidents_train_09.columns:

#     print(column)

    train_10[column] = np.nan

    

train_10 = train_10.join(fin_incidents_train_09, on='USER_ID', rsuffix='_other')

train_10.fillna(0, inplace=True)
model, y_pred = fit_and_pred_logreg(train_10, test)
interim_test['PREDICT'] = y_pred

interim_test[['USER_ID', 'PREDICT']].to_csv('baseline_submission.csv', index=False) # В папке output. Выгрузить ручками