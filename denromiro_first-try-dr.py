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



x_cleared = x.fillna(x.median(0)).copy(deep=True)

x_cleared_check_na = x_cleared.isna().any()

x_cleared = x_cleared.dropna(axis=1)
x_check_na = x.isna().any()

x_summary = x.describe()

x_summary = x_summary.T



x_cleared = x.fillna(x.median(0)).copy(deep=True)

x_cleared_check_na = x_cleared.isna().any()

x_cleared = x_cleared.dropna(axis=1)



# clearing autoregression

x_cleared1 = x_cleared.corr()

x_cleared1.reset_index(inplace=True)

x_cleared2 = pd.melt(x_cleared1, id_vars=['index'])

x_cleared2 = x_cleared2[x_cleared2['index']!=x_cleared2['variable']]

x_cleared2 = x_cleared2.sort_values(by=['value'], ascending=False)



x_cleared2 = x_cleared2[abs(x_cleared2['value'])>0.85]

x_cleared2 = x_cleared2.iloc[::2, :]



x_cleared.drop(x_cleared2['index'], axis=1, inplace=True)



x_cleared1 = x_cleared.corr()

x_cleared1.reset_index(inplace=True)

x_cleared2 = pd.melt(x_cleared1, id_vars=['index'])

x_cleared2 = x_cleared2[x_cleared2['index']!=x_cleared2['variable']]

x_cleared2 = x_cleared2.sort_values(by=['value'], ascending=False)



x_check_na = x_cleared.isna().any()

x_cleared = x_cleared.drop('id', axis='columns') 
x_train = x_cleared.iloc[:4000, :]

x_test = x_cleared.iloc[4000:, :]
#from sklearn.ensemble import RandomForestClassifier



#rnd_clf = RandomForestClassifier(max_depth=100,

#                                 n_estimators=1000, 

#                                 max_features = 0.9,

#                                 max_leaf_nodes=1000, 

#                                 min_samples_leaf = 1,                                    

#                                 n_jobs=-1, 

#                                 verbose=1, 

#                                 random_state=1)



#rnd_clf.fit(x_train, y)



#sub['covid_19'] = rnd_clf.predict_proba(x_test)[:,1]
#sub.head()
#sub.to_csv("first_run.csv", index=False)
x_train1 = x_train.loc[:, (x_train != 0).any(axis=0)].copy(deep=True)

#x_train1['Lipase dosage'] = pd.to_numeric(x_train1['Lipase dosage'])



x_test1 = x_test.loc[:,list(x_train1.columns)]

#x_test1['Lipase dosage'] = pd.to_numeric(x_test1['Lipase dosage'])







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
sub.to_csv("second_run.csv", index=False)