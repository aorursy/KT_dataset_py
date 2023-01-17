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
train = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/Train.csv')

test = pd.read_csv("/kaggle/input/janatahack-healthcare-analytics/Test.csv")

submit = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/sample_submmission.csv')

print(train.shape, test.shape)
patient = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/Patient_Profile.csv')

camp = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/Health_Camp_Detail.csv')

hc1 = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/First_Health_Camp_Attended.csv')

hc2 = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/Second_Health_Camp_Attended.csv')

hc3 = pd.read_csv('/kaggle/input/janatahack-healthcare-analytics/Train/Third_Health_Camp_Attended.csv')
train.head(2)
train.isnull().sum()
train['pat_camp'] = (train['Patient_ID'].astype(str) + '_' + train['Health_Camp_ID'].astype(str)).astype(str)

train.head(2)

test['pat_camp'] = (test['Patient_ID'].astype(str) + '_' + test['Health_Camp_ID'].astype(str)).astype(str)

test.head(2)
print(f'Shape of train: {train.shape}')

for col in list(train.columns):

    print(f'Distinct entries in {col}: {train[col].nunique()}')
train.info()
train[train['Registration_Date'].isnull()].head()
print('Missing registration date info:')

print(f'Missing rows: {train["Registration_Date"].isnull().sum()}')

print(f'Unique patients related to missing info: {train[train["Registration_Date"].isnull()]["Patient_ID"].nunique()}')

print(f'Unique health camps related to missing info: {train[train["Registration_Date"].isnull()]["Health_Camp_ID"].nunique()}')
train['Registration_Date'] = pd.to_datetime(train['Registration_Date'])

test['Registration_Date'] = pd.to_datetime(test['Registration_Date'])
print(np.min(train['Registration_Date']), np.max(train['Registration_Date']))

print(np.min(test['Registration_Date']), np.max(test['Registration_Date']))
patient.head(2)
patient.isnull().sum()
patient.info()
print(f'Shape of patient: {patient.shape}')

for col in list(patient.columns):

    print(f'Distinct entries in {col}: {patient[col].nunique()}')
print(f'Distinct entries in City_Type: {patient.City_Type.unique()}')

print(f'Distinct entries in Employer_Category: {patient.Employer_Category.unique()}')
patient.City_Type = patient.City_Type.fillna('None')

patient.Employer_Category = patient.Employer_Category.fillna('None')
patients = patient.Patient_ID.nunique()

assert patients == patient.shape[0]



for col in list(patient.columns):

    print(f'None type entries %age in {col}: {100*(patient[patient[col]=="None"][col].count()/patients)}')
patient[patient['Income'] != "None"]['Income'].astype(int).hist()
patient[patient['Age'] != "None"]['Age'].astype(float).hist()
patient[patient['Education_Score'] != "None"]['Education_Score'].astype(float).hist()
patient['First_Interaction'] = pd.to_datetime(patient['First_Interaction'])

patient.head(2)
patient['Income'] = [ np.nan if y == "None" else y for y in patient['Income']]

patient['Age'] = [ np.nan if y == "None" else y for y in patient['Age']]

patient['Education_Score'] = [ np.nan if y == "None" else y for y in patient['Education_Score']]



patient['Income'] = (patient['Income'].astype(float)).fillna(patient['Income'].mode())

patient['Age'] = patient['Age'].astype(float)

patient['Age'] = patient['Age'].fillna(np.mean(patient['Age']))

patient['Education_Score'] = patient['Education_Score'].astype(float)

patient['Education_Score'] = patient['Education_Score'].fillna(np.mean(patient['Age']))
hc1['pat_camp'] = (hc1['Patient_ID'].astype(str) + '_' + hc1['Health_Camp_ID'].astype(str)).astype(str)

hc2['pat_camp'] = (hc2['Patient_ID'].astype(str) + '_' + hc2['Health_Camp_ID'].astype(str)).astype(str)

hc3['pat_camp'] = (hc3['Patient_ID'].astype(str) + '_' + hc3['Health_Camp_ID'].astype(str)).astype(str)

hc1.head(2)
hc3.Number_of_stall_visited.value_counts()
print(len(np.intersect1d(hc1.Patient_ID, hc2.Patient_ID)))

print(len(np.intersect1d(hc1.Patient_ID, hc3.Patient_ID)))

print(len(np.intersect1d(hc2.Patient_ID, hc3.Patient_ID)))

print(len(np.intersect1d(np.intersect1d(hc1.Patient_ID, hc2.Patient_ID), hc3.Patient_ID)))
print(len(np.intersect1d(hc1.Health_Camp_ID, hc2.Health_Camp_ID)))

print(len(np.intersect1d(hc1.Health_Camp_ID, hc3.Health_Camp_ID)))

print(len(np.intersect1d(hc2.Health_Camp_ID, hc3.Health_Camp_ID)))
hc1 = hc1[['Patient_ID', 'Health_Camp_ID', 'Donation', 'Health_Score', 'pat_camp']]

hc1.head(2)
temp = ['Health_Score' if col == 'Health Score' else col for col in hc2.columns]

hc2.columns = temp

del temp

hc2['Donation'] = 0

hc2 = hc2[hc1.columns]

hc2.head(2)
hc3.head(2)
print(train.shape[0], hc1.shape[0] + hc2.shape[0] + hc3.shape[0])

print(len(np.intersect1d(train.pat_camp, hc1.pat_camp)))

print(len(np.intersect1d(train.pat_camp, hc2.pat_camp)))

print(len(np.intersect1d(train.pat_camp, hc3.pat_camp)))
camp.head(5)
camp['Camp_Start_Date'] = pd.to_datetime(camp['Camp_Start_Date'])

camp['Camp_End_Date'] = pd.to_datetime(camp['Camp_End_Date'])

camp.head()
print(f'Distinct entries in Category1: {camp.Category1.unique()}')

print(f'Distinct entries in Category2: {camp.Category2.unique()}')

print(f'Distinct entries in Category3: {camp.Category3.unique()}')
camp['duration'] = [divmod((camp['Camp_End_Date'].iloc[x]-camp['Camp_Start_Date'].iloc[x]).total_seconds(), 86400)[0]+1 for x in camp.index]

camp.head(2)
np.log(camp['duration']).hist()
camp['duration'] = np.log(camp['duration'])
train = pd.merge(train, camp, on = 'Health_Camp_ID', how = 'left')

train.head(2)
test = pd.merge(test, camp, on = 'Health_Camp_ID', how = 'left')

test.head(2)
train =pd.merge(train, patient, on = 'Patient_ID', how = 'left')

train.head(2) 

test = pd.merge(test, patient, on = 'Patient_ID', how = 'left')

test.head(2)
pat = list(hc1['pat_camp']) + list(hc2['pat_camp']) + list(hc3[hc3['Number_of_stall_visited']>0]['pat_camp'])

out = pd.DataFrame(pat, columns = ['pat_camp'])

out['outcome'] = 1

print(out.shape)
assert len(np.intersect1d(out.pat_camp, train.pat_camp)) == out.shape[0]

out.head(2)
train = pd.merge(train, out, on = 'pat_camp', how = 'left')

train['outcome'] = train.outcome.fillna(0)

train.head()
train['enrol_days'] = [divmod((train['Registration_Date'].iloc[x]-train['First_Interaction'].iloc[x]).total_seconds(), 86400)[0] +1 for x in train.index]

test['enrol_days'] = [divmod((test['Registration_Date'].iloc[x]-test['First_Interaction'].iloc[x]).total_seconds(), 86400)[0] +1 for x in test.index]
train['enrol_days'] = train['enrol_days'].fillna(train['enrol_days'].mode())

test['enrol_days'] = test['enrol_days'].fillna(test['enrol_days'].mode())
train.enrol_days.hist()
train.enrol_days = np.log(train.enrol_days)

test.enrol_days = np.log(test.enrol_days)
train['days_since_camp_start'] = [divmod((train['Registration_Date'].iloc[x]-train['Camp_Start_Date'].iloc[x]).total_seconds(), 86400)[0] +1 for x in train.index]

test['days_since_camp_start'] = [divmod((test['Registration_Date'].iloc[x]-test['Camp_Start_Date'].iloc[x]).total_seconds(), 86400)[0] +1 for x in test.index]
train['days_duration_ratio'] = train['days_since_camp_start']/train['duration']

test['days_duration_ratio'] = test['days_since_camp_start']/test['duration']
'''

train['days_since_camp_start'] = np.log(train['days_since_camp_start'])

test['days_since_camp_start'] = np.log(test['days_since_camp_start'])



train['days_duration_ratio'] = np.log(train['days_duration_ratio'])

train['days_duration_ratio'] = np.log(train['days_duration_ratio'])

'''
train['days_for_camp_end'] = [divmod((train['Camp_End_Date'].iloc[x]-train['Registration_Date'].iloc[x]).total_seconds(), 86400)[0] +1 for x in train.index]

test['days_for_camp_end'] = [divmod((test['Camp_End_Date'].iloc[x]-test['Registration_Date'].iloc[x]).total_seconds(), 86400)[0] +1 for x in test.index]



train['enddays_duration_ratio'] = train['days_for_camp_end']/train['duration']

test['enddays_duration_ratio'] = test['days_for_camp_end']/test['duration']



'''

train['days_for_camp_end'] = np.log(train['days_for_camp_end'])

test['days_for_camp_end'] = np.log(test['days_for_camp_end'])



train['enddays_duration_ratio'] = np.log(train['enddays_duration_ratio'])

test['enddays_duration_ratio'] = np.log(test['enddays_duration_ratio'])

'''
test['outcome'] = -1

data = train.append(test)
new_df = pd.DataFrame() 

new_df['total_visits'] = np.log(data.groupby('Patient_ID')['Health_Camp_ID'].count())



new_df['total_first'] = data[data['Category1'] == 'First'].groupby('Patient_ID')['Health_Camp_ID'].count() 

new_df['total_second'] = data[data['Category1'] == 'Second'].groupby('Patient_ID')['Health_Camp_ID'].count() 

new_df['total_third'] = data[data['Category1'] == 'Third'].groupby('Patient_ID')['Health_Camp_ID'].count()



new_df['total_first'] = new_df['total_first'].fillna(0) 

new_df['total_second'] = new_df['total_second'].fillna(0) 

new_df['total_third'] = new_df['total_third'].fillna(0)



new_df = new_df.reset_index() 

print(new_df.head())



data = pd.merge(data, new_df, on = 'Patient_ID', how = 'left')
new_df = pd.DataFrame()

new_df['total_patients'] = np.log(data.groupby('Health_Camp_ID')['Patient_ID'].count())

data = pd.merge(data, new_df, on = 'Health_Camp_ID', how = 'left')
data.info()
min_reg = np.min(data['Registration_Date'])

data['Registration_Date'] = [divmod((x-min_reg).total_seconds(), 86400)[0] +1 for x in data['Registration_Date']]



min_start = np.min(data['Camp_Start_Date'])

data['Camp_Start_Date'] = [divmod((x-min_start).total_seconds(), 86400)[0] +1 for x in data['Camp_Start_Date']]

min_end = np.min(data['Camp_End_Date'])

data['Camp_End_Date'] = [divmod((x-min_end).total_seconds(), 86400)[0] +1 for x in data['Camp_End_Date']]



min_inter = np.min(data['First_Interaction'])

data['First_Interaction'] = [divmod((x-min_inter).total_seconds(), 86400)[0] +1 for x in data['First_Interaction']]
hc3.head()
new_df = data[['Patient_ID', 'Registration_Date', 'outcome']].sort_values(['Patient_ID', 'Registration_Date']).reset_index(drop=True)



pat_list = []

last_outcome = []

for i, pat_id in enumerate(list(new_df.Patient_ID.unique())):

    if(i%10000 ==0):

        print(i)

    temp = new_df[new_df['Patient_ID'] == pat_id].reset_index(drop=True)

    temp2 = temp['outcome'][temp.shape[0]-1]

    pat_list.append(pat_id)

    last_outcome.append(temp2)
new_df = pd.DataFrame.from_dict({'Patient_ID': pat_list, 'last_outcome': last_outcome})

new_df.head()

"""

new_df = pd.DataFrame()



temp = hc1[['Health_Score', 'Patient_ID']].append(hc2[['Health_Score', 'Patient_ID']])

temp.columns = ['Health_Score', 'Patient_ID']

new_df['mean_score'] = temp.groupby('Patient_ID')['Health_Score'].mean()



temp = hc1[['Donation', 'Patient_ID']]

temp.columns = ['Donation', 'Patient_ID']

new_df['sum_donation'] = temp.groupby('Patient_ID')['Donation'].sum()

new_df['count_donation'] = temp.groupby('Patient_ID')['Donation'].count()



new_df['sum_donation'] = new_df['sum_donation'].fillna(0)

new_df['count_donation'] = new_df['count_donation'].fillna(0)



temp = hc3[['Number_of_stall_visited', 'Patient_ID']]

temp.columns = ['Number_of_stall_visited', 'Patient_ID']

new_df['total_stalls_visited'] = temp.groupby('Patient_ID')['Number_of_stall_visited'].sum()

new_df['mean_stalls_visited'] = temp.groupby('Patient_ID')['Number_of_stall_visited'].mean()

new_df['total_stalls_visited'] = new_df['total_stalls_visited'].fillna(0)

new_df['mean_stalls_visited'] = new_df['mean_stalls_visited'].fillna(0)



new_df = new_df.reset_index()

new_df.head()

"""
data = pd.merge(data, new_df, on = 'Patient_ID', how = 'left')

data[new_df.columns] = data[new_df.columns].fillna(0)

data.head()
assert len(list(train.columns)) == len(list(test.columns)) 
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from category_encoders import TargetEncoder, MEstimateEncoder

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
cat_cols = ['Category1', 'Category2', 'Category3','City_Type','Employer_Category']



for col in cat_cols:

    print(col)

    encoder.fit(data[col])

    data[col] = encoder.transform(data[col]).astype('float')
train2 = data[data['outcome'] != -1]

test2 = data[data['outcome'] == -1]
X = train2.drop(columns = ['Patient_ID',

                          'Health_Camp_ID'

                          , 'pat_camp'

                          ,'Online_Follower', 'LinkedIn_Shared'

                          , 'Twitter_Shared', 'Facebook_Shared'

                           , 'Var4', 'Var2', 'Var3'

                           #'Registration_Date', 'Camp_Start_Date', 'Camp_End_Date'

                          , 'Income','Education_Score', 'Age'

                          #, 'First_Interaction'

                           , 'outcome', 'last_outcome'

                         ])

y = train2['outcome']

print(X.shape, y.shape)
Xtest = test2.drop(columns = ['Patient_ID',

                            'Health_Camp_ID'

                          , 'pat_camp'

                          ,'Online_Follower', 'LinkedIn_Shared'

                          , 'Twitter_Shared', 'Facebook_Shared'

                          , 'Var4', 'Var2', 'Var3'

                          #'Registration_Date', 'Camp_Start_Date', 'Camp_End_Date'

                          , 'Income','Education_Score', 'Age'

                          #, 'First_Interaction'

                           , 'outcome', 'last_outcome'

                         ])
print(X.shape, Xtest.shape)
X.head()
import lightgbm as lgb

from lightgbm import LGBMClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, f1_score, make_scorer
params = {}

params['boosting_type']= 'dart' #dropout aided regressive trees (DART) # improves accuracy

params['learning_rate']= 0.05

params['verbose']: 0

    

#params["objective"] = "binary:logistic"

params['metric'] = 'auc'

params["min_data_in_leaf"] = 8 

params["bagging_fraction"] = 0.7

params["feature_fraction"] = 0.7

params["bagging_seed"] = 50



model = LGBMClassifier(objective = 'binary', boosting_type= 'dart', learning_rate = 0.05, metric = 'auc', num_estimators = 600

                       , random_state = 22, min_data_in_leaf = 8, bagging_fraction = 0.7, feature_fraction = 0.7)

cv = RepeatedKFold(n_splits = 5, n_repeats = 1, random_state = 22)

n_scores = cross_val_score(model, X, y, scoring = make_scorer(roc_auc_score), cv = cv )

print(np.mean(n_scores), n_scores)



model = XGBClassifier(objective = 'binary:logistic',boosting_type= 'dart', learning_rate = 0.05, metric = 'auc', num_estimators = 600

                       , random_state = 22, min_data_in_leaf = 8, bagging_fraction = 0.7, feature_fraction = 0.7)

cv = RepeatedKFold(n_splits = 5, n_repeats = 1, random_state = 22)

n_scores = cross_val_score(model, X, y, scoring = make_scorer(roc_auc_score), cv = cv )



#print(roc_auc_score(model.predict(Xtrain), yval))

print(np.mean(n_scores))
def runLGB2(Xtrain, ytrain, Xval, yval, cat_cols, Xtest = None):

    params = {

    'boosting_type': 'dart', #dropout aided regressive trees (DART) # improves accuracy

    #'max_depth': 10, 

    'learning_rate': 0.05

    ,'verbose': 1

    }

    

    #regularising for overfitting with inf depth

    params["objective"] = "binary"

    params['metric'] = 'auc'

    params["min_data_in_leaf"] = 8 

    params["bagging_fraction"] = 0.7

    params["feature_fraction"] = 0.7

    params["bagging_freq"] = 3

    params["bagging_seed"] = 50



    n_estimators = 1000

    early_stopping_rounds = 10



    d_train = lgb.Dataset(Xtrain.copy(), label=ytrain.copy(), categorical_feature=cat_cols)

    d_valid = lgb.Dataset(Xval.copy(), label=yval.copy(), categorical_feature=cat_cols)

    watchlist = [d_train, d_valid]



    model = lgb.train(params, d_train, n_estimators

                      , watchlist

                      , verbose_eval=80

                      , early_stopping_rounds=early_stopping_rounds)



    preds = model.predict(Xval, num_iteration=model.best_iteration)

    err = roc_auc_score(yval, preds)

    

    preds_test = model.predict(Xtest, num_iteration=model.best_iteration)

    return  preds, err, preds_test, model
def runXGB(train_X, train_y, test_X, test_y=None,  extra_X=None, num_rounds=200):

	params = {}

	params["objective"] = "binary:logistic"

	params['eval_metric'] = 'auc'

	params["eta"] = 0.02 

	params["subsample"] = 0.8

	params["min_child_weight"] = 5

	params["colsample_bytree"] = 0.7

	params["max_depth"] = 6

	params["silent"] = 1

	params["seed"] = 0



	plst = list(params.items())

	xgtrain = xgb.DMatrix(train_X, label=train_y)



	if test_y is not None:

		xgtest = xgb.DMatrix(test_X, label=test_y)

		watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

		model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=300)

	else:

		xgtest = xgb.DMatrix(test_X)

		model = xgb.train(plst, xgtrain, num_rounds)



	pred_test_y = model.predict(xgtest)

	loss = 0

    

	if extra_X is not None:

		xgtest = xgb.DMatrix(extra_X)

		pred_extra_y = model.predict(xgtest)

		return pred_test_y, pred_extra_y, loss, model 



	if test_y is not None:

		loss = roc_auc_score(test_y, pred_test_y)

		print (loss)

		return pred_test_y, loss, model

	else:

	    return pred_test_y,loss, model
y = pd.DataFrame(y)

y['Category1'] = X['Category1'].copy()

y.head()
import warnings

warnings.filterwarnings("ignore")



import time

cat_cols2 = cat_cols# + ['last_outcome']



df =pd.DataFrame()

for camp_format in X.Category1.unique():

    preds_buff = 0

    err_buff = []



    n_splits = 5

    kf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=22)

   

    for dev_index, val_index in kf.split(X[X['Category1'] == camp_format], y[y['Category1'] == camp_format]['outcome'].astype(int)):

        start = time.time()

        Xtrain, Xval = X[X['Category1'] == camp_format].iloc[dev_index], X[X['Category1'] == camp_format].iloc[val_index]

        ytrain, yval = y[y['Category1'] == camp_format]['outcome'].astype(int).iloc[dev_index], y[y['Category1'] == camp_format]['outcome'].astype(int).iloc[val_index]    

        

        print(Xtrain.shape, Xval.shape, ytrain.shape, yval.shape)

        

        pred_val, roc, preds_test, model = runLGB2(Xtrain, ytrain, Xval, yval, cat_cols2, Xtest[Xtest['Category1'] == camp_format])

        preds_buff += preds_test

        err_buff.append(roc)

        print(f'Mean Error: {np.mean(err_buff)}; Split error: {roc}')

        print(f'Total time in seconds for this fold: {time.time()-start}')

        print('\n')

    '''



    Xtrain, Xval, ytrain, yval = train_test_split(X[X['Category1'] == camp_format], y[y['Category1'] == camp_format]['outcome'].astype(int), test_size = 0.2, random_state = 22, stratify = y[y['Category1'] == camp_format]['outcome'])

    print(Xtrain.shape, Xval.shape, ytrain.shape, yval.shape)

    pred_val, roc, preds_test, model = runLGB2(Xtrain, ytrain, Xval, yval, cat_cols2, Xtest[Xtest['Category1'] == camp_format])

    print(roc)

    '''

    

    temp_df = test2[test2['Category1'] == camp_format][['Patient_ID', 'Health_Camp_ID']].copy()

    temp_df['Outcome'] = preds_buff/n_splits

    df = df.append(temp_df)
df.head()
df.to_csv('lgb_v2.csv', index = False)
Xtrain, Xval, ytrain, yval = train_test_split(X, y['outcome'], test_size = 0.2, random_state = 22, stratify = y)

pred_val, roc, preds_test, model = runLGB2(Xtrain, ytrain, Xval, yval, cat_cols2, Xtest)

roc
#submit['Outcome'] = model.predict(Xtest, num_iteration=model.best_iteration)

#submit[['Patient_ID', 'Health_Camp_ID','Outcome']].to_csv('lgb_v7.csv', index = False)
submit['Outcome'] = preds_test

submit.to_csv('gb_1000_v1.csv', index = False)
submit.head()
a =model.feature_importance(importance_type='split')

feature = pd.DataFrame(model.feature_name())

feature['impo'] = a

feature = feature.sort_values(by = ['impo'], ascending = False)

feature.head(30)
X.isnull().sum()
import time



preds_buff = 0

err_buff = []



n_splits = 5

kf = StratifiedKFold(n_splits=n_splits, shuffle= True, random_state=22)



for dev_index, val_index in kf.split(X, y['outcome']):

    start = time.time()

    Xtrain, Xval = X.iloc[dev_index], X.iloc[val_index]

    ytrain, yval = np.array(y['outcome'].iloc[dev_index]), np.array(y['outcome'].iloc[val_index])

    

    pred_val, roc, preds_test, model = runLGB2(Xtrain, ytrain, Xval, yval, cat_cols, Xtest)

    preds_buff += preds_test

    err_buff.append(roc)

    print(f'Mean Error: {np.mean(err_buff)}; Split error: {roc}')

    print(f'Total time in seconds for this fold: {time.time()-start}')

    print('\n')



preds_buff /= n_splits
print(err_buff, np.mean(err_buff))
submit['Outcome'] = preds_buff

submit.to_csv('gb_gbdt_new_v4.csv', index = False)
print(len(np.intersect1d(test.pat_camp, hc1.pat_camp)))

print(len(np.intersect1d(test.pat_camp, hc2.pat_camp)))

print(len(np.intersect1d(test.pat_camp, hc3.pat_camp)))
print(len(np.intersect1d(train[train['outcome'] == 1].Patient_ID.unique(), test.Patient_ID)))

print(len(np.intersect1d(train[train['outcome'] == 1].Health_Camp_ID.unique(), test.Health_Camp_ID)))
test['Patient_ID'].nunique()