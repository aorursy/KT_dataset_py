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
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



import xgboost as xgb
import matplotlib.pyplot as plt

plt.style.use('classic')
df = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv', index_col='Id')

df.info()
df_nans = df.isna().sum()[df.isna().sum() >0]

df_nans
df[df_nans.index].info()
df[df_nans.index].describe()
fig, axs = plt.subplots(1,5,figsize=(24,8))



for idx, col in enumerate(df_nans.index):

    axs[idx].set_title(col)

    axs[idx].boxplot(df[col].dropna(axis=0))



plt.show()
def fill_nas(df):

    df['v2a1'] = df['v2a1'].fillna(df['v2a1'].median())

    df['meaneduc'] = df['meaneduc'].fillna(df['meaneduc'].median())

    df['SQBmeaned'] = df['SQBmeaned'].fillna(df['SQBmeaned'].median())

    df['v18q1'] = df['v18q1'].fillna(-1)

    df['rez_esc'] = df['rez_esc'].fillna(-1)

    return df
df = fill_nas(df)
df.isna().sum()[df.isna().sum() >0]
df.select_dtypes(include=['object'])
df['edjefe'].unique()
df['edjefa'].unique()
df['dependency'].unique()
def replace_yes_no(df, column):

    df['{}_yes'.format(column)] = df[column].apply(lambda row: 1 if row=='yes' else 0)

    df['{}_no'.format(column)] = df[column].apply(lambda row: 1 if row=='no' else 0)

    df[column] = df[column].apply(lambda row: row if row not in ['yes', 'no'] else -1)

    df[column] = pd.to_numeric(df[column])

    return df
def replace_yes_no_all(df):

    df = replace_yes_no(df, 'edjefe')

    df = replace_yes_no(df, 'edjefa')

    df = replace_yes_no(df, 'dependency')

    return df
df = replace_yes_no_all(df)
fig, axs = plt.subplots(1,3,figsize=(24,8))



axs[0].hist(df['dependency'])

axs[1].hist(df['edjefe'])

axs[2].hist(df['edjefa'])



plt.show()

df.select_dtypes(include=['float64'])
df['Target']
# add the number of people over 18 in each household

def add_over_18(df):

    df['num_over_18'] = 0

    df['num_over_18'] = df[df.age >= 18].groupby('idhogar').transform("count")

    df['num_over_18'] = df.groupby("idhogar")["num_over_18"].transform("max")

    df['num_over_18'] = df['num_over_18'].fillna(0)

    return df



df = add_over_18(df)

# add some extra features, these were taken from another kernel

def extract_features(df):

    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']

    df['rent_to_rooms'] = df['v2a1']/df['rooms']

    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household

    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household

    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household

    df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3'] # rent to people in household

    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1']) # rent to people under age 12

    df['hhsize_to_rooms'] = df['hhsize']/df['rooms'] # rooms per person

    df['rent_to_hhsize'] = df['v2a1']/df['hhsize'] # rent to household size

    df['rent_to_over_18'] = df['v2a1']/df['num_over_18']

    # some households have no one over 18, use the total rent for those

    df.loc[df.num_over_18 == 0, "rent_to_over_18"] = df[df.num_over_18 == 0].v2a1

    return df

    

df = extract_features(df) 
X = df.drop(['Target', 'idhogar'], axis=1)

y = df['Target']
y = y.apply(lambda row: row-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
fit_params={"eval_metric" : 'merror', 

            "eval_set" : [(X_train,y_train), (X_test, y_test)],

           }
xgb_model = xgb.XGBClassifier(n_jobs=4)

xgb_model.fit(X_train, y_train, **fit_params)
y_pred = xgb_model.predict(X_test)
y_pred
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred, average='macro')
rf_model = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=42)

rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

confusion_matrix(y_test, y_rf_pred)
accuracy_score(y_test, y_rf_pred)
f1_score(y_test, y_rf_pred, average='macro')
gb_model = GradientBoostingClassifier(random_state=42)

gb_model.fit(X_train, y_train)
y_gb_pred = gb_model.predict(X_test)

confusion_matrix(y_test, y_gb_pred)
accuracy_score(y_test, y_gb_pred)
f1_score(y_test, y_gb_pred, average='macro')
df_submit = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv', index_col='Id')
df_submit.isna().sum()[df_submit.isna().sum() >0]
df_submit_cleaned = df_submit
df_submit_cleaned = replace_yes_no_all(df_submit_cleaned)

df_submit_cleaned = fill_nas(df_submit_cleaned)

df_submit_cleaned = add_over_18(df_submit_cleaned)

df_submit_cleaned = extract_features(df_submit_cleaned)
df_submit_cleaned = df_submit.drop(['idhogar'], axis=1)
x_submit = df_submit_cleaned
y_submit_raw = xgb_model.predict(x_submit)
y_submit = pd.DataFrame(y_submit_raw, index=df_submit_cleaned.index, columns=['Target'])
y_submit['Target'].unique()
y_submit['Target'] = y_submit['Target'].apply(lambda row: row+1)

y_submit['Target'].unique()
y_submit.to_csv('submission.csv')