!python --version
import pkg_resources

dists = [d for d in pkg_resources.working_set]

print(dists)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

sub=pd.read_csv("../input/predicttheglobalspreadofcovid19/SampleSubmission.csv")

sub.head()

new=sub['Territory X Date'].str.split(" X ", n = 1, expand = True) 

sub["Country_Region"]=new[0]

sub["Date"]=new[1]



# test=pd.read_csv("../input/covid19globalforecastingweek4/test.csv")

test = pd.read_csv('../input/predicttheglobalspreadofcovid19/new_test.csv')

test.head()

test2=test.groupby(['Province_State','Country_Region']).size().reset_index()

test2.columns=['Province_State','Country_Region','Count']

sub=sub.merge(test2,how="left",on="Country_Region")

sub.drop(['Count'],1,inplace=True)

sub['Date'] = pd.to_datetime(sub['Date'], format='%m/%d/%y')

sub['ForecastId']=sub.index+1

sub.head()
%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 99)

import os

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm

import datetime as dt

import xgboost as xgb

from sklearn import preprocessing

from scipy.stats import gmean
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))



import plotly.express as px

import plotly.graph_objects as go


DATA_PATH = '../input/covid19-metadata'

DATEFORMAT = '%Y-%m-%d'





def get_comp_data(COMP):

    train = pd.read_csv(f'{COMP}/train.csv')

    test = sub.copy()#pd.read_csv(f'{COMP}/test.csv')

    test['Date']=test['Date'].astype(str)

    submission = pd.read_csv(f'{COMP}/submission.csv')

    print(train.shape, test.shape, submission.shape)

    train['Country_Region'] = train['Country_Region'].str.replace(',', '')

    test['Country_Region'] = test['Country_Region'].str.replace(',', '')



    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')



    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')



    train['LogConfirmed'] = to_log(train.ConfirmedCases)

    train['LogFatalities'] = to_log(train.Fatalities)

    train = train.drop(columns=['Province_State'])

    test = test.drop(columns=['Province_State'])



    country_codes = pd.read_csv(f'{DATA_PATH}/country_codes.csv', keep_default_na=False)

    train = train.merge(country_codes, on='Country_Region', how='left')

    test = test.merge(country_codes, on='Country_Region', how='left')



    train['DateTime'] = pd.to_datetime(train['Date'])

    test['DateTime'] = pd.to_datetime(test['Date'])



    train = train.sort_values(by='Date')

    test = test.sort_values(by='Date')



    train = train.fillna('#N/A')

    test = test.fillna('#N/A')



    return train, test, submission





def process_each_location(df):

    dfs = []

    for loc, df in df.groupby('Location'):

        df = df.sort_values(by='Date')

        df['Fatalities'] = df['Fatalities'].cummax()

        df['ConfirmedCases'] = df['ConfirmedCases'].cummax()

        df['LogFatalities'] = df['LogFatalities'].cummax()

        df['LogConfirmed'] = df['LogConfirmed'].cummax()

        df['LogConfirmedNextDay'] = df['LogConfirmed'].shift(-1)

        df['ConfirmedNextDay'] = df['ConfirmedCases'].shift(-1)

        df['DateNextDay'] = df['Date'].shift(-1)

        df['LogFatalitiesNextDay'] = df['LogFatalities'].shift(-1)

        df['FatalitiesNextDay'] = df['Fatalities'].shift(-1)

        df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']

        df['ConfirmedDelta'] = df['ConfirmedNextDay'] - df['ConfirmedCases']

        df['LogFatalitiesDelta'] = df['LogFatalitiesNextDay'] - df['LogFatalities']

        df['FatalitiesDelta'] = df['FatalitiesNextDay'] - df['Fatalities']

        dfs.append(df)

    return pd.concat(dfs)





def add_days(d, k):

    return dt.datetime.strptime(d, DATEFORMAT) + dt.timedelta(days=k)





def to_log(x):

    return np.log(x + 1)





def to_exp(x):

    return np.exp(x) - 1





def create_features(train_set):

    dfs = []

    for loc, df in train_set.groupby('Location'):

        df = df.sort_values(by='Date').copy()

        df['f_lc_7d'] = df['LogConfirmed'].shift(7)

        df['f_lf_7d'] = df['LogFatalities'].shift(7)

        df['f_lc_3d'] = df['LogConfirmed'].shift(3)

        df['f_lf_3d'] = df['LogFatalities'].shift(3)

        df['f_lc_1d'] = df['LogConfirmed'].shift(1)

        df['f_lf_1d'] = df['LogFatalities'].shift(1)

        df['f_lc_0d'] = df['LogConfirmed']

        df['f_lf_0d'] = df['LogFatalities']

        df['f_fc_rate'] = np.clip((to_exp(df['LogFatalities']) + 1) / (to_exp(df['LogConfirmed']) + 1), 0, 0.15)

        dfs.append(df)

    dfs = pd.concat(dfs)

    dfs['d_lc_7d'] = dfs['f_lc_0d'] - dfs['f_lc_7d']

    dfs['d_lf_7d'] = dfs['f_lf_0d'] - dfs['f_lf_7d']

    dfs['d_lc_3d'] = dfs['f_lc_3d'] - dfs['f_lc_3d']

    dfs['d_lf_3d'] = dfs['f_lf_3d'] - dfs['f_lf_3d']

    dfs['d_lc_1d'] = dfs['f_lc_0d'] - dfs['f_lc_1d']

    dfs['d_lf_1d'] = dfs['f_lf_0d'] - dfs['f_lf_1d']

    return dfs



COMP = '../input/covid19globalforecastingweek4'

start = dt.datetime.now()

train, test, submission = get_comp_data(COMP)

train.shape, test.shape, submission.shape

train.head(2)

test.head(2)



TRAIN_START = train.Date.min()

TEST_START = test.Date.min()

TRAIN_END = train.Date.max()

TEST_END = test.Date.max()

print(TRAIN_START, TRAIN_END, TEST_START, TEST_END)
train_clean = process_each_location(train)



print('train cleaned', train_clean.shape)



train_clean = train_clean[[

    'Location', 'Date', 'continent',

    'LogConfirmed', 'LogFatalities',

    'LogConfirmedDelta', 'LogFatalitiesDelta'

]]



continent_encoder = preprocessing.LabelEncoder()

train_clean['f_continent'] = continent_encoder.fit_transform(train_clean.continent)



train_features = create_features(train_clean)

train_features.head()
def predict(min_child_weight, eta, colsample_bytree, max_depth, subsample,

           NROUND, PRECISION, DECAY, WEIGHT_NORM, MIN_DATE):

    train, test, submission = get_comp_data(COMP)

    train_clean = process_each_location(train)

    print('train cleaned', train_clean.shape)



    train_clean = train_clean[[

        'Location', 'Date', 'continent',

        'LogConfirmed', 'LogFatalities',

        'LogConfirmedDelta', 'LogFatalitiesDelta'

    ]]



    continent_encoder = preprocessing.LabelEncoder()

    train_clean['f_continent'] = continent_encoder.fit_transform(train_clean.continent)



    train_features = create_features(train_clean)



    VALID_CUTOFF = TRAIN_END



    features = [

        'f_lc_7d', 'f_lf_7d', 'f_lc_3d', 'f_lf_3d', 'f_continent',

        'f_lc_1d', 'f_lf_1d', 'f_lc_0d', 'f_lf_0d', 'f_fc_rate'

    ]

    diff_features = ['d_lc_7d', 'd_lf_7d', 'd_lc_3d', 'd_lf_3d', 'd_lc_1d', 'd_lf_1d']

    features = features + diff_features

    print(f'{len(features)} features: ')

    print(features)

    

    fix = {

    'lambda': 1., 'nthread': 3, 'booster': 'gbtree',

    'silent': 1, 'eval_metric': 'rmse',

    'objective': 'reg:squarederror'}

    config = dict(

        min_child_weight=min_child_weight,

        eta=eta, colsample_bytree=colsample_bytree,

        max_depth=max_depth, subsample=subsample)

    config.update(fix)



    Xtr = train_features[(train_features.Date >= MIN_DATE) & (train_features.Date < VALID_CUTOFF)].copy()

    Xtr['days'] = -(pd.to_datetime(train_features.Date) - dt.datetime.strptime(VALID_CUTOFF, DATEFORMAT)).dt.days

    

    print(Xtr.shape)

    print(config)

    print(PRECISION, NROUND, DECAY, WEIGHT_NORM)



    def weighting(days):

        return 1. / days ** WEIGHT_NORM



    dtrain_lc = xgb.DMatrix(Xtr[features].round(PRECISION), label=Xtr.LogConfirmedDelta, weight=weighting(Xtr.days))

    dtrain_lf = xgb.DMatrix(Xtr[features].round(PRECISION), label=Xtr.LogFatalitiesDelta, weight=weighting(Xtr.days))



    model_lc = xgb.train(config, dtrain_lc, NROUND, evals=[(dtrain_lc, 'train-lc')], verbose_eval=100)

    model_lf = xgb.train(config, dtrain_lf, NROUND, evals=[(dtrain_lf, 'train-lf')], verbose_eval=100)



    # Predict



    predictions = Xtr.copy()

    predictions = train_features[(train_features.Date >= MIN_DATE) & (train_features.Date <= VALID_CUTOFF)].copy()

    predictions.LogConfirmedDelta = np.nan

    predictions.LogFatalitiesDelta = np.nan



    for i, d in enumerate(pd.date_range(VALID_CUTOFF, add_days(TEST_END, 1))):

        last_day = str(d).split(' ')[0]

        next_day = dt.datetime.strptime(last_day, DATEFORMAT) + dt.timedelta(days=1)

        next_day = next_day.strftime(DATEFORMAT)



        p_next_day = predictions[predictions.Date == last_day].copy()

        p_next_day.Date = next_day

        p_next_day['plc'] = model_lc.predict(xgb.DMatrix(p_next_day[features].round(PRECISION)))

        p_next_day['plf'] = model_lf.predict(xgb.DMatrix(p_next_day[features].round(PRECISION)))



        p_next_day.LogConfirmed = p_next_day.LogConfirmed + np.clip(p_next_day['plc'], 0, None) * DECAY ** i

        p_next_day.LogFatalities = p_next_day.LogFatalities + np.clip(p_next_day['plf'], 0, None) * DECAY ** i



        predictions = pd.concat([predictions, p_next_day], sort=True)

        predictions = create_features(predictions)



    predictions['PC'] = to_exp(predictions.LogConfirmed)

    predictions['PF'] = to_exp(predictions.LogFatalities)

    return predictions
decay = 0.99

prediction_1 = predict(min_child_weight=5, eta=0.01, colsample_bytree=0.8, max_depth=5, subsample=0.9,

           NROUND=800, PRECISION=2, DECAY=decay, WEIGHT_NORM=0.25, MIN_DATE='2020-03-22')

prediction_2 = predict(min_child_weight=7, eta=0.01, colsample_bytree=0.7, max_depth=6, subsample=0.8,

           NROUND=1000, PRECISION=2, DECAY=decay, WEIGHT_NORM=0.23, MIN_DATE='2020-03-15')

prediction_3 = predict(min_child_weight=3, eta=0.01, colsample_bytree=0.6, max_depth=7, subsample=0.7,

           NROUND=1200, PRECISION=2, DECAY=decay, WEIGHT_NORM=0.2, MIN_DATE='2020-03-08')

prediction_4 = predict(min_child_weight=3, eta=0.011, colsample_bytree=0.75, max_depth=10, subsample=0.6,

           NROUND=1200, PRECISION=3, DECAY=decay, WEIGHT_NORM=0.15, MIN_DATE='2020-03-15')

prediction_5 = predict(min_child_weight=10, eta=0.008, colsample_bytree=0.75, max_depth=10, subsample=0.6,

           NROUND=1500, PRECISION=3, DECAY=decay, WEIGHT_NORM=0.2, MIN_DATE='2020-03-22')

prediction_6 = predict(min_child_weight=20, eta=0.01, colsample_bytree=0.7, max_depth=5, subsample=0.85,

           NROUND=1000, PRECISION=3, DECAY=decay, WEIGHT_NORM=0.3, MIN_DATE='2020-03-22')
cols = ['Location', 'Date', 'PC', 'PF']

p12 = pd.merge(prediction_1[cols], prediction_2[cols], on=['Location', 'Date'], suffixes=['_1', '_2'])

p34 = pd.merge(prediction_3[cols], prediction_4[cols], on=['Location', 'Date'], suffixes=['_3', '_4'])

p56 = pd.merge(prediction_5[cols], prediction_6[cols], on=['Location', 'Date'], suffixes=['_5', '_6'])



preds = pd.merge(p12, p34, on=['Location', 'Date'])

preds = pd.merge(preds, p56, on=['Location', 'Date'])

preds.head()



c = preds.loc[preds.Date >= '2020-04-15'].corr()

c

fig = px.imshow(c)

fig.show()
pcs = ['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6']

pfs = ['PF_1', 'PF_2', 'PF_3', 'PF_4', 'PF_5', 'PF_6']

preds['PC'] = to_exp(to_log(preds[pcs]).mean(axis=1))

preds['PF'] = to_exp(to_log(preds[pfs]).mean(axis=1))

preds.tail()
top_locations = preds[preds.Date == TRAIN_END].sort_values(by='PF', ascending=False).Location.values[:25]

fig3 = px.line(preds[preds.Location.isin(top_locations)],

               x='Date', y='PC', color='Location')

_ = fig3.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Predicted Cumulative Confirmed Cases by Location [Updated: {TRAIN_END}]'

)

fig3.show()
top_locations = preds[preds.Date == TRAIN_END].sort_values(by='PF', ascending=False).Location.values[:25]

fig3 = px.line(preds[preds.Location.isin(top_locations)],

               x='Date', y='PF', color='Location')

_ = fig3.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Predicted Cumulative Deaths by Location [Updated: {TRAIN_END}]'

)

fig3.show()
total = preds.groupby('Date')[['PC', 'PF'] + pcs + pfs].sum().reset_index()

total.tail()

fig2 = px.line(pd.melt(total, id_vars=['Date']), x='Date', y='value', color='variable')

_ = fig2.update_layout(

    yaxis_type="log",

    title_text=f'COVID-19 Cumulative Prediction Total [Updated: {TRAIN_END}]'

)

fig2.show()
my_submission = test.copy()

my_submission = my_submission.merge(preds)

my_submission['ConfirmedCases'] = my_submission['PC']

my_submission['Fatalities'] = my_submission['PF']

my_submission.shape

my_submission.head()
my_submission[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)

my_submission.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index().tail()
end = dt.datetime.now()

print('Finished', end, (end - start).seconds, 's')
my_submission['target']=my_submission['Fatalities'].fillna(0)

my_submission.sort_values(by='Territory X Date',inplace=True)

my_submission[['Territory X Date','target']].to_csv("submission_beluga.csv",index=False)

my_submission[['Territory X Date','target']].tail()
my_submission[['Territory X Date','target']].head()
sub=pd.read_csv("submission.csv")

sub.head()
test = pd.read_csv('../input/predicttheglobalspreadofcovid19/new_test.csv')

test=test.merge(sub,on="ForecastId",how="left")

test.to_csv('submissionBELUGA.csv', index=None)

test.head()
best=pd.read_csv("submissionBELUGA.csv")

best['Country_Region'].value_counts()
sub=pd.read_csv("../input/predicttheglobalspreadofcovid19/SampleSubmission.csv")

sub.shape
new=sub['Territory X Date'].str.split(" X ", n = 1, expand = True) 

sub["Country_Region"]=new[0]

sub["Date"]=new[1]

sub['Date'] = pd.to_datetime(sub['Date'], format='%m/%d/%y')

sub['Date']= sub['Date'].astype(str)

print(sub.shape)

sub.head()
print(best.shape)

best.head()
best2=best.groupby(['Country_Region','Date'])['Fatalities'].sum().reset_index()

best2.loc[best2['Country_Region']=="US",'Country_Region']="United States of America (the)"

best2.loc[best2['Country_Region']=="Russia",'Country_Region']="Russian Federation (the)"

best2.loc[best2['Country_Region']=="United Kingdom",'Country_Region']="United Kingdom of Great Britain and Northern Ireland (the)"

best2.loc[best2['Country_Region']=="Iran",'Country_Region']="Iran (Islamic Republic of)"

best2.loc[best2['Country_Region']=="Venezuela",'Country_Region']="Venezuela (Bolivarian Republic of)"

best2.loc[best2['Country_Region']=="United Arab Emirates",'Country_Region']="United Arab Emirates (the)"

best2.loc[best2['Country_Region']=="Gambia",'Country_Region']="Gambia (the)"

best2.loc[best2['Country_Region']=="Netherlands",'Country_Region']="Netherlands (the)"

best2.loc[best2['Country_Region']=="Niger",'Country_Region']="Niger (the)"

best2.loc[best2['Country_Region']=="Tanzania",'Country_Region']="United Republic of Tanzania (the)"

best2.loc[best2['Country_Region']=="Vietnam",'Country_Region']="Viet Nam"

best2.loc[best2['Country_Region']=="Bahamas",'Country_Region']="Bahamas (the)"

best2.loc[best2['Country_Region']=="Bolivia",'Country_Region']="Bolivia (Plurinational State of)"

best2.loc[best2['Country_Region']=="Central African Republic",'Country_Region']="Central African Republic (the)"

best2.loc[best2['Country_Region']=="Congo",'Country_Region']="Congo (the)"

best2.loc[best2['Country_Region']=="Comoros",'Country_Region']="Comoros (the)"

best2.loc[best2['Country_Region']=="Korea, South",'Country_Region']="Republic of Korea (the)"

best2.loc[best2['Country_Region']=="Democratic Republic of the Congo",'Country_Region']="Democratic Republic of the Congo (the)"

best2.loc[best2['Country_Region']=="Dominican Republic",'Country_Region']="Dominican Republic (the)"

best2.loc[best2['Country_Region']=="Moldova",'Country_Region']="Republic of Moldova (the)"

best2.loc[best2['Country_Region']=="Sudan",'Country_Region']="Sudan (the)"

best2.loc[best2['Country_Region']=="Syria",'Country_Region']="Syrian Arab Republic (the)"

best2.loc[best2['Country_Region']=="Philippines",'Country_Region']="Philippines (the)"

best2.loc[best2['Country_Region']=="Congo (Kinshasa)",'Country_Region']="Democratic Republic of the Congo (the)"

best2.loc[best2['Country_Region']=="Congo (Brazzaville)",'Country_Region']="Congo (the)"

best2.loc[best2['Country_Region']=="Cote d'Ivoire",'Country_Region']="Côte d'Ivoire"

best2.loc[best2['Country_Region']=="Laos",'Country_Region']="Lao People's Democratic Republic (the)"

sub=sub.merge(best2,how="left",on=['Country_Region','Date'])

sub['target']=sub['Fatalities'].fillna(0)

print(sub.shape)

sub.head()
sub[['Territory X Date','target']].to_csv("ma20200417_beluga.csv",index=False)

print(sub[['Territory X Date','target']].shape)

sub[['Territory X Date','target']].head()
a=sub.groupby(['Date'])['Fatalities'].sum().reset_index()

a[30:50]
# sub1=pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

# print(sub1.shape)

# sub2=pd.read_csv("../input/covid19globalforecastingweek4/submission.csv")

# print(sub2.shape)

# train=pd.read_csv("../input/covid19globalforecastingweek4/train.csv")

# print(train.shape)

# train.head()
# test=pd.read_csv("../input/covid19globalforecastingweek4/test.csv")

# print(test.shape)

# test.head()
# best=pd.read_csv("submission_beluga.csv")

# print(best.shape)

# best.head()
# best=pd.read_csv("submission.csv")

# print(best.shape)

# best.head()
# best=best.merge(test,how="left",on="ForecastId")

# print(best.shape)

# best.head()
# sub=pd.read_csv("../input/predicttheglobalspreadofcovid19/SampleSubmission.csv")

# print(sub.shape)

# sub.head()
# new=sub['Territory X Date'].str.split(" X ", n = 1, expand = True) 

# sub["Country_Region"]=new[0]

# sub["Date"]=new[1]

# sub['Date'] = pd.to_datetime(sub['Date'], format='%m/%d/%y')

# sub['Date']= sub['Date'].astype(str)

# sub.head()

# best2=best.groupby(['Country_Region','Date'])['Fatalities'].sum().reset_index()

# best2.loc[best2['Country_Region']=="US",'Country_Region']="United States of America (the)"

# best2.loc[best2['Country_Region']=="Russia",'Country_Region']="Russian Federation (the)"

# best2.loc[best2['Country_Region']=="United Kingdom",'Country_Region']="United Kingdom of Great Britain and Northern Ireland (the)"

# best2.loc[best2['Country_Region']=="Iran",'Country_Region']="Iran (Islamic Republic of)"

# best2.loc[best2['Country_Region']=="Venezuela",'Country_Region']="Venezuela (Bolivarian Republic of)"

# best2.loc[best2['Country_Region']=="United Arab Emirates",'Country_Region']="United Arab Emirates (the)"

# best2.loc[best2['Country_Region']=="Gambia",'Country_Region']="Gambia (the)"

# best2.loc[best2['Country_Region']=="Netherlands",'Country_Region']="Netherlands (the)"

# best2.loc[best2['Country_Region']=="Niger",'Country_Region']="Niger (the)"

# best2.loc[best2['Country_Region']=="Tanzania",'Country_Region']="United Republic of Tanzania (the)"

# best2.loc[best2['Country_Region']=="Vietnam",'Country_Region']="Viet Nam"

# best2.loc[best2['Country_Region']=="Bahamas",'Country_Region']="Bahamas (the)"

# best2.loc[best2['Country_Region']=="Bolivia",'Country_Region']="Bolivia (Plurinational State of)"

# best2.loc[best2['Country_Region']=="Central African Republic",'Country_Region']="Central African Republic (the)"

# best2.loc[best2['Country_Region']=="Congo",'Country_Region']="Congo (the)"

# best2.loc[best2['Country_Region']=="Comoros",'Country_Region']="Comoros (the)"

# best2.loc[best2['Country_Region']=="Korea, South",'Country_Region']="Republic of Korea (the)"

# best2.loc[best2['Country_Region']=="Democratic Republic of the Congo",'Country_Region']="Democratic Republic of the Congo (the)"

# best2.loc[best2['Country_Region']=="Dominican Republic",'Country_Region']="Dominican Republic (the)"

# best2.loc[best2['Country_Region']=="Moldova",'Country_Region']="Republic of Moldova (the)"

# best2.loc[best2['Country_Region']=="Sudan",'Country_Region']="Sudan (the)"

# best2.loc[best2['Country_Region']=="Syria",'Country_Region']="Syrian Arab Republic (the)"

# best2.head()
# sub=sub.merge(best2,how="left",on=['Country_Region','Date'])

# sub['target']=sub['Fatalities'].fillna(0)

# sub[['Territory X Date','target']].to_csv("ma20200417_beluga.csv",index=False)

# sub[['Territory X Date','target']].head()
# print(sub[['Territory X Date','target']].shape)
# sub[['Territory X Date','target']].tail()