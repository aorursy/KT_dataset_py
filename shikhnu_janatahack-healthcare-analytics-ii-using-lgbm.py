import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
train_data = pd.read_csv('../input/janatahack-healthcare-ii/JanataHack_HealthCare_II_train.csv')
test_data = pd.read_csv('../input/janatahack-healthcare-ii/JanataHack_HealthCare_II_test.csv')
train_data.head()
train_data.columns = train_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test_data.columns = test_data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
train_data.head()
print('Train Data Shape: ', train_data.shape)
print('Test Data Shape: ', test_data.shape)
train_data.head()
train_data.dtypes
train_data.isnull().sum()
train_data.nunique()
train_data.columns
# Unique values for all the columns
for col in train_data.columns[~(train_data.columns.isin(['case_id', 'patientid', 'admission_deposit']))].tolist():
    print(" Unique Values --> " + col, ':', len(train_data[col].unique()), ': ', train_data[col].unique())
i = 1
for column in train_data.columns[~(train_data.columns.isin(['case_id', 'patientid', 'admission_deposit']))].tolist():
    plt.figure(figsize = (60, 10))
    plt.subplot(4, 4, i)
    sns.barplot(x = train_data[column].value_counts().index, y = train_data[column].value_counts())
    i += 1
    plt.show()
sns.boxplot(x = 'visitors_with_patient', data = train_data)
sns.despine()
plt.figure(figsize = (20, 6))
sns.barplot(x = train_data.groupby(['severity_of_illness'])['visitors_with_patient'].value_counts().index, y = train_data.groupby(['severity_of_illness'])['visitors_with_patient'].value_counts())
plt.xticks(rotation = 90)
sns.despine()
train_data = train_data.fillna('NaN')
test_data = test_data.fillna('NaN')

for column in train_data.columns[~(train_data.columns.isin(['case_id', 'stay']))].tolist():

    le = LabelEncoder()

    if column == 'city_code_patient':
        train_data['city_code_patient'] = train_data['city_code_patient'].astype('str')
        test_data['city_code_patient'] = test_data['city_code_patient'].astype('str')
        train_data['city_code_patient'] = le.fit_transform(train_data['city_code_patient'])
        test_data['city_code_patient'] = le.fit_transform(test_data['city_code_patient'])
    
    elif column == 'bed_grade':
        bedGrade = {1: '1',2: '2', 3: '3', 4: '4', np.nan: '5'}
        train_data['bed_grade'] = train_data['bed_grade'].map(bedGrade)
        test_data['bed_grade'] = test_data['bed_grade'].map(bedGrade)
        train_data['bed_grade'] = train_data['bed_grade'].fillna('NaN')
        test_data['bed_grade'] = test_data['bed_grade'].fillna('NaN')
    
    else:
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.fit_transform(test_data[column])
train_data.head()
train_data.shape
ss = StandardScaler()

for column in train_data.columns[~(train_data.columns.isin(['case_id', 'stay']))].tolist():
    train_data[[column]] = ss.fit_transform(train_data[[column]])
    test_data[[column]] = ss.fit_transform(test_data[[column]])
train_data.head()
train_data.dropna(inplace=True)
X=train_data.drop(['stay','case_id'],axis=1).astype('float32')
y=train_data['stay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
import lightgbm as lgb
lgbc = lgb.LGBMClassifier(random_state=1)

lgbc.fit(X_train, y_train)

y_pred_test = lgbc.predict(X_test)
y_prob_test = lgbc.predict_proba(X_test)[:,1]

y_pred_train = lgbc.predict(X_train)
y_prob_train = lgbc.predict_proba(X_train)[:,1]

print('Accuracy on Train Set: ', accuracy_score(y_train, y_pred_train))
print('Accuracy on Test Set: ', accuracy_score(y_test, y_pred_test))
# Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

lgbc = lgb.LGBMClassifier(random_state=1)

params = {'n_estimators': sp_randint(5,10),
          'max_depth' : sp_randint(2,10),
          'min_child_samples' : sp_randint(1,20),
          'num_leaves' : sp_randint(5,10)}

rand_search_lgbc = RandomizedSearchCV(lgbc, param_distributions=params, random_state=1, cv=3)

rand_search_lgbc.fit(X_train, y_train)

rand_search_lgbc.best_params_
# Passing best parameter for the Hyperparameter Tuning
lgbc_ht = lgb.LGBMClassifier(**rand_search_lgbc.best_params_, random_state=1)

lgbc_ht.fit(X_train, y_train)

y_pred_test = lgbc_ht.predict(X_test)
y_prob_test = lgbc_ht.predict_proba(X_test)[:,1]

y_pred_train = lgbc_ht.predict(X_train)
y_prob_train = lgbc_ht.predict_proba(X_train)[:,1]

print('Accuracy on Train Set: ', accuracy_score(y_train, y_pred_train))
print('Accuracy on Test Set: ', accuracy_score(y_test, y_pred_test))
test_data = test_data.fillna(method='ffill')
lgbc.fit(X_train, y_train)

predictions = lgbc.predict(test_data[test_data.columns[~(test_data.columns.isin(['case_id']))].tolist()].values)
submission = pd.DataFrame({'case_id': test_data['case_id'], 'Stay': predictions.ravel()})
submission.to_csv('LGBM_predict.csv', index = False)
submission.head()
