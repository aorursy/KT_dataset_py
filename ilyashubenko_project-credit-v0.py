# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import shapiro
from scipy.stats import probplot
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import warnings
warnings.simplefilter('ignore')

TRAIN_DATASET_PATH = '/kaggle/input/credit-default/train.csv'
TEST_DATASET_PATH = '/kaggle/input/credit-default/test.csv'
df_train = pd.read_csv(TRAIN_DATASET_PATH) # загружаем тренировочный датасет в датафрейм df_train
df_test = pd.read_csv(TEST_DATASET_PATH) # загружаем тестовый датасет в датафрейм df_test
TARGET_NAME = 'Credit Default'
BASE_FEATURE_NAMES = df_train.columns.drop(TARGET_NAME).tolist()
#NEW_FEATURE_NAMES = ['']
NUMB_FEATURE_NAMES = ['Annual Income','Tax Liens','Number of Open Accounts','Years of Credit History',
                     'Maximum Open Credit','Number of Credit Problems','Months since last delinquent',
                     'Bankruptcies','Current Loan Amount','Current Credit Balance','Monthly Debt','Credit Score']
CAT_FEATURE_NAMES = ['Years in current job','Home Ownership','Purpose','Term']
# Целевая переменная
y = df_train[[TARGET_NAME]]
y.info()
plt.figure(figsize=(8, 5))

sns.countplot(x=TARGET_NAME, data=df_train)

plt.title('Target variable distribution')
plt.show()
df_train.head(10)
corr_with_target = df_train.corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()
plt.figure(figsize=(10, 8))

sns.countplot(x="Years in current job", hue=TARGET_NAME, data=df_train)
plt.title('\"Years in current job\" grouped by target variable')
plt.legend(title='Target', loc='upper right')

plt.show()
# пропуски
df_train.isna().sum()
# Добавляю фичу "имеет задержку" и присваиваю всем наблюдениям 1
df_train['has_delay'] = 1

# Заменяю NaN на 0 в 'Months since last delinquent'
df_train['Months since last delinquent'].fillna(0, inplace=True)

# У кого нет задолженности - ставим 0 в новый признак
df_train.loc[(df_train['Months since last delinquent'] == 0), 'has_delay'] = 0
# Добавляю фичу об неизвестном доходе и присваиваю всем наблюдениям 0
df_train['unknown_income'] = 0

# В переменную annual_income_median записываю мелианное значение зарплат
annual_income_median = df_train['Annual Income'].median()

# Всем, у кого неизвестна зарплата делаем пометку "1" в столбец unknown_income
df_train.loc[(df_train['Annual Income'].isnull()), 'unknown_income'] = 1

# заполняем пропуски зарплат медианным значением
df_train['Annual Income'].fillna(annual_income_median, inplace=True)

df_train.isnull().sum()
df_train

NEW_FEATURE_NAMES = ['has_delay','unknown_income']
corr_with_target = df_train[BASE_FEATURE_NAMES + 
                            NEW_FEATURE_NAMES + 
                            [TARGET_NAME]].corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()
# Сколько людей, в группах по стажу работы, не возвращает кредит

g = sns.catplot("Credit Default", col="Years in current job", col_wrap=5,
                data=df_train,
                kind="count", height=3.5, aspect=.8, 
                palette='tab20')

#fig.suptitle('sf')
plt.show()

plt.figure(figsize = (10, 3))

df_train['Credit Score'].hist(bins=30, )
plt.ylabel('Count')
plt.xlabel('Credit Score')

plt.title('bla bla')
plt.show()
df_train.isnull().sum()
# Добавляю фичу об неизвестном Credit Score и присваиваю всем наблюдениям 0
df_train['unknown_credit_score'] = 0
NEW_FEATURE_NAMES = ['has_delay','unknown_income','unknown_credit_score']

# В переменную credit_score_median записываю медианное значение рейтинга
credit_score_median = df_train['Credit Score'].median()

# Всем, у кого неизвестен кредитный рейтинг делаем пометку "1" в столбец unknown_credit_score
df_train.loc[(df_train['Credit Score'].isnull()), 'unknown_credit_score'] = 1

# заполняем пропуски рейтинга медианным значением
df_train['Credit Score'].fillna(credit_score_median, inplace=True)

import random

# Уникальные значения поля Years in current job
unique_years_in_current_job = df_train['Years in current job'].unique()

# Срез, чтобы убрать NaN
var_experiance = unique_years_in_current_job[1:]

# Заменяем NaN на рандомный опыт
df_train['Years in current job'].fillna(random.choice(var_experiance), inplace=True)
df_train.isnull().sum()
df_train
# значения 99999999.0 поля Current Loan Amount меняем на медианные
median_current_loan_amount = df_train['Current Loan Amount'].median()
df_train.loc[(df_train['Current Loan Amount'] == 99999999.0), 'Current Loan Amount'] = median_current_loan_amount
df_train.Bankruptcies.value_counts()
#mode_bankruptcies = df_train['Bankruptcies'].mode()
df_train['Bankruptcies'].fillna(0 , inplace=True)
df_train.Bankruptcies.value_counts()
df_train.isnull().sum()
for cat_colname in df_train.select_dtypes(include='object').columns:
    print(str(cat_colname) + '\n\n' + str(df_train[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')
#for cat_colname in df_train.select_dtypes(include='object').columns[1:]:
#    df = pd.concat([df_train, pd.get_dummies(df_train[cat_colname], prefix=cat_colname)], axis=1)
df_train['term_binary'] = df_train['Term'].map({'Short Term':'1', 'Long Term':'0'}).astype(int)
NEW_FEATURE_NAMES = ['has_delay','unknown_income','unknown_credit_score','term_binary']
df_train.select_dtypes(include='object').columns[1:]
#df_train["Home Ownership"] = df_train["Home Ownership"].astype("category")
#df_train = pd.get_dummies(df_train)
#df_train.head()

#df['IS_MALE'] = df['SEX'].map({'1':'1', '2':'0'}).astype(int)
#df_train = 
corr_with_target = df_train[BASE_FEATURE_NAMES + 
                            NEW_FEATURE_NAMES + 
                            [TARGET_NAME]].corr().iloc[:-1, -1].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

sns.barplot(x=corr_with_target.values, y=corr_with_target.index)

plt.title('Correlation with target variable')
plt.show()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, learning_curve
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb, lightgbm as lgbm, catboost as catb
def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_test_true, y_test_pred))
def balance_df_by_target(df, target_name):

    target_counts = df[target_name].value_counts()

    major_class_name = target_counts.argmax()
    minor_class_name = target_counts.argmin()

    disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1

    for i in range(disbalance_coeff):
        sample = df[df[target_name] == minor_class_name].sample(target_counts[minor_class_name])
        df = df.append(sample, ignore_index=True)

    return df.sample(frac=1) 

df_train
NEW_FEATURE_NAMES
SELECTED_FEATURE_NAMES = NUMB_FEATURE_NAMES + NEW_FEATURE_NAMES


X = df_train[SELECTED_FEATURE_NAMES]
y = df_train[TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=11)
scaler = StandardScaler()

df_norm = df_train.copy()
df_norm[NUMB_FEATURE_NAMES] = scaler.fit_transform(df_norm[NUMB_FEATURE_NAMES])

df_train = df_norm.copy()
df_for_balancing = pd.concat([X_train, y_train], axis=1)
df_balanced = balance_df_by_target(df_for_balancing, TARGET_NAME)
    
df_balanced[TARGET_NAME].value_counts()
X_train = df_balanced.drop(columns=TARGET_NAME)
y_train = df_balanced[TARGET_NAME]
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

y_train_pred = model_lr.predict(X_train)
y_test_pred = model_lr.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

y_train_pred = model_knn.predict(X_train)
y_test_pred = model_knn.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_xgb = xgb.XGBClassifier(random_state=11)
model_xgb.fit(X_train, y_train)

y_train_pred = model_xgb.predict(X_train)
y_test_pred = model_xgb.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_lgbm = lgbm.LGBMClassifier(random_state=11)
model_lgbm.fit(X_train, y_train)

y_train_pred = model_lgbm.predict(X_train)
y_test_pred = model_lgbm.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_catb = catb.CatBoostClassifier(silent=True, random_state=11)
model_catb.fit(X_train, y_train)

y_train_pred = model_catb.predict(X_train)
y_test_pred = model_catb.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
model_catb = catb.CatBoostClassifier(class_weights=[1, 3.5], silent=True, random_state=11)
params = {'n_estimators':[50, 100, 200, 500, 700, 1000, 1200, 1500],
          'max_depth':[3, 5, 7]}
cv=KFold(n_splits=3, random_state=11, shuffle=True)
%%time

rs = RandomizedSearchCV(model_catb, params, scoring='f1', cv=cv, n_jobs=-1)
rs.fit(X, y)
rs.best_params_
rs.best_score_
%%time

final_model = catb.CatBoostClassifier(n_estimators=1500, max_depth=3,
                                      silent=True, random_state=11)
final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

get_classification_report(y_train, y_train_pred, y_test, y_test_pred)
# Добавляю фичу "имеет задержку" и присваиваю всем наблюдениям 1
df_test['has_delay'] = 1

# Заменяю NaN на 0 в 'Months since last delinquent'
df_test['Months since last delinquent'].fillna(0, inplace=True)

# У кого нет задолженности - ставим 0 в новый признак
df_test.loc[(df_test['Months since last delinquent'] == 0), 'has_delay'] = 0

# Добавляю фичу об неизвестном доходе и присваиваю всем наблюдениям 0
df_test['unknown_income'] = 0

# В переменную annual_income_median записываю мелианное значение зарплат
annual_income_median = df_test['Annual Income'].median()

# Всем, у кого неизвестна зарплата делаем пометку "1" в столбец unknown_income
df_test.loc[(df_test['Annual Income'].isnull()), 'unknown_income'] = 1

# заполняем пропуски зарплат медианным значением
df_test['Annual Income'].fillna(annual_income_median, inplace=True)

# Добавляю фичу об неизвестном Credit Score и присваиваю всем наблюдениям 0
df_test['unknown_credit_score'] = 0
NEW_FEATURE_NAMES = ['has_delay','unknown_income','unknown_credit_score']

# В переменную credit_score_median записываю медианное значение рейтинга
credit_score_median = df_test['Credit Score'].median()

# Всем, у кого неизвестен кредитный рейтинг делаем пометку "1" в столбец unknown_credit_score
df_test.loc[(df_test['Credit Score'].isnull()), 'unknown_credit_score'] = 1

# заполняем пропуски рейтинга медианным значением
df_test['Credit Score'].fillna(credit_score_median, inplace=True)

# Уникальные значения поля Years in current job
unique_years_in_current_job = df_test['Years in current job'].unique()

# Срез, чтобы убрать NaN
var_experiance = unique_years_in_current_job[1:]

# Заменяем NaN на рандомный опыт
df_test['Years in current job'].fillna(random.choice(var_experiance), inplace=True)

# значения 99999999.0 поля Current Loan Amount меняем на медианные
median_current_loan_amount = df_test['Current Loan Amount'].median()
df_test.loc[(df_test['Current Loan Amount'] == 99999999.0), 'Current Loan Amount'] = median_current_loan_amount

df_test['term_binary'] = df_test['Term'].map({'Short Term':'1', 'Long Term':'0'}).astype(int)
df_for_model = df_test[SELECTED_FEATURE_NAMES]
df_for_model
y_pred_output = final_model.predict(df_for_model)
y_pred_output.shape
SAMPLE = '/kaggle/input/credit-default/sample_submission.csv'
df_sample = pd.read_csv(SAMPLE)
df_sample
predictions = pd.DataFrame(index = df_sample['Id'])

# predictions = df_sample(index = df_sample['Id'])
# predictions = df_sample.drop(columns=TARGET_NAME)
# predictions.index = predictions['Id']
#predictions = pd.DataFrame(index = df_test.index)
predictions
# X_test.isnull().sum()
predictions[TARGET_NAME] = y_pred_output
predictions
predictions.to_csv('out_credit_3.csv', sep=',')
#!kaggle competitions submit -c credit-default -m "ILYA Shubenko" -f try_test_1.csv