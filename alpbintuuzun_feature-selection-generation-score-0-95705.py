

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn import metrics

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched'])

#Printing kickstarter data columns

list(data.columns)


list(data.state.unique())


data = data.query('state != "live"')

data = data.assign(isSuccessful = (data['state']=='successful').astype(int))

nop = len(data)

nos = len(data.query('isSuccessful == 1'))

ros = nos/nop

print("Number of projects: ", nop, " number of successful projects: " ,nos, " ratio of success: ", ros)

data = data.assign(lhour=data.launched.dt.hour,

               lday=data.launched.dt.day,

               lmonth=data.launched.dt.month,

               lyear=data.launched.dt.year)
data = data[(data['launched'] >= '2009-04-28')]
data = data.assign(dhour=data.deadline.dt.hour,

               dday=data.deadline.dt.day,

               dmonth=data.deadline.dt.month,

               dyear=data.deadline.dt.year)
data = data.assign(total_year= data.dyear-data.lyear)

data = data.assign(total_month = data.total_year*12 + (data.dmonth-data.lmonth))

data = data.assign(total_day = data.total_month * 30 + (data.dday - data.lday))

data = data.assign(total_hour = data.total_day * 24 + (data.dhour - data.lhour))
data = data.query('total_hour >= 0')

data
low = data['goal'].quantile(0.3)

mid = data['goal'].quantile(0.6)

high = data['goal'].quantile(0.9)



conditions = [(data['goal']<low),((data['goal']>=low) & (data['goal']<mid)),((data['goal']>=mid) & (data['goal']<high)),(data['goal']>=high)]

tags = ['t0_low', 't1_medium','t2_high','t3_extreme']



data = data.assign(categorical_goal = np.select(conditions,tags))

low = data['backers'].quantile(0.3)

mid = data['backers'].quantile(0.6)

high = data['backers'].quantile(0.9)





conditions = [(data['backers']<low),((data['backers']>=low) & (data['backers']<mid)),((data['backers']>=mid) & (data['backers']<high)),(data['backers']>=high)]

tags = ['t0_low', 't1_medium','t2_high','t3_extreme']



data = data.assign(categorical_backers = np.select(conditions,tags))



data.head()
groupableData = data.drop(columns = ['ID', 'name','goal','backers', 'deadline','dyear','lyear','dmonth','lmonth','dday','lday','dhour','lhour','launched','usd_goal_real','pledged','state','usd_pledged_real','usd pledged'])

groupableData
mainCategorySuccess = groupableData[['main_category','isSuccessful']]

mainCategorySuccess = mainCategorySuccess.groupby(['main_category']).mean().sort_values(by=['isSuccessful'], ascending = False)

mainCategorySuccess
categorySuccess = groupableData[['category','isSuccessful']]

categorySuccess = categorySuccess.groupby(['category']).mean().sort_values(by=['isSuccessful'], ascending = False)

categorySuccess
currencySuccess = groupableData[['currency','isSuccessful']]

currencySuccess = currencySuccess.groupby(['currency']).mean()

currencySuccess
countrySuccess = groupableData[['country','isSuccessful']]

countrySuccess = countrySuccess.groupby(['country']).mean()

countrySuccess
goalSuccess = groupableData[['categorical_goal','isSuccessful']]

goalSuccess = goalSuccess.groupby(['categorical_goal']).mean()

goalSuccess.plot()
backerSuccess = groupableData[['categorical_backers','isSuccessful']]

backerSuccess = backerSuccess.groupby(['categorical_backers']).mean()

backerSuccess.plot()
yearSuccess = groupableData[['total_year','isSuccessful']]

yearSuccess = yearSuccess.groupby(['total_year']).mean()

yearSuccess.plot()
monthSuccess = groupableData[['total_month','isSuccessful']]

monthSuccess = monthSuccess.groupby(['total_month']).mean()

monthSuccess.plot()
daySuccess = groupableData[['total_day','isSuccessful']]

daySuccess = daySuccess.groupby(['total_day']).mean()

daySuccess.plot()
hourSuccess = groupableData[['total_hour','isSuccessful']]

hourSuccess = hourSuccess.groupby(['total_hour']).mean()

hourSuccess.plot()
encoder = LabelEncoder()

data = groupableData.apply(encoder.fit_transform)



data
frac = 0.1

size = int(len(data)*frac)



trainSet = data[:-2*size]

validationSet = data[-2*size: -size]

testSet = data[-size:]
features = trainSet.columns.drop('isSuccessful')



dtrain = lgb.Dataset(trainSet[features], label=trainSet['isSuccessful'])

dvalid = lgb.Dataset(validationSet[features], label=validationSet['isSuccessful'])



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 1000

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(testSet[features])

score = metrics.roc_auc_score(testSet['isSuccessful'], ypred)



print(f"Test AUC score: {score}")