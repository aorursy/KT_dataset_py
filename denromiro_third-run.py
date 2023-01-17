import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_train.csv")

test = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_test.csv")

sub = pd.read_csv("/kaggle/input/covid-diagnostic/covid_19_submission.csv")
y = train.covid_19.values

train_test = train.drop("covid_19", axis='columns')

x = pd.concat([train_test, test])



x_check_na = x.isna().any()

x_summary = x.describe()

x_summary = x_summary.T



# df = df.loc[:, df.isnull().mean() < .8]

x_cleared = x.loc[:, x.isnull().mean() < 0.9]

# x_checknull =  x_cleared.isnull().mean()



x_cleared = x_cleared.fillna(x.median(0)).copy(deep=True)

x_check_na = x_cleared.isna().any()



x_cleared1 = x_cleared.corr()

x_cleared1.reset_index(inplace=True)

x_cleared2 = pd.melt(x_cleared1, id_vars=['index'])

x_cleared2 = x_cleared2[x_cleared2['index']!=x_cleared2['variable']]

x_cleared2 = x_cleared2.sort_values(by=['value'], ascending=False)



x_cleared2 = x_cleared2[abs(x_cleared2['value'])>0.85]

x_cleared2 = x_cleared2.iloc[::2, :]



x_cleared.drop(x_cleared2['index'], axis=1, inplace=True)

x_cleared = x_cleared.drop('id', axis='columns')



x_cleared = x_cleared.apply(pd.to_numeric, errors = 'coerce') 

x_check_na = x_cleared.isna().any()



x_train1 = x_cleared.iloc[:4000, :]

x_test1 = x_cleared.iloc[4000:, :]



import xgboost as xgb



xg_class = xgb.XGBClassifier(

    learning_rate=0.02, 

    max_delta_step=0, 

    max_depth=10,

    min_child_weight=0.1, 

    missing=None, 

    n_estimators=250, 

    nthread=4,

    objective='binary:logistic', 

    reg_alpha=0.01, 

    reg_lambda = 0.01,

    scale_pos_weight=1, 

    seed=0, 

    silent=False, 

    subsample=0.9)



xg_fit=xg_class.fit(x_train1, y)



sub['covid_19'] = xg_class.predict_proba(x_test1)[:,1]
sub.head()
sub.to_csv("third_run.csv", index=False)