import numpy as np

import pandas as pd

import datetime as dt

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.sandbox.tools import cross_val
train2017 = pd.read_csv('../input/train_2017.csv')

train2016 = pd.read_csv('../input/train_2016_v2.csv')

prop = pd.read_csv('../input/properties_2016.csv', low_memory = False)

prop17 = pd.read_csv('../input/properties_2017.csv', low_memory = False)
zip_count = prop['regionidzip'].value_counts().to_dict()

city_count = prop['regionidcity'].value_counts().to_dict()

medyear = prop.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()

meanarea = prop.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()

medlat = prop.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()

medlong = prop.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()



zip_count17 = prop17['regionidzip'].value_counts().to_dict()

city_count17 = prop17['regionidcity'].value_counts().to_dict()

medyear17 = prop17.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()

meanarea17 = prop17.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()

medlat17 = prop17.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()

medlong17 = prop17.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()
train2016 = train2016.merge(prop, how='left', on=['parcelid'])

train2017 = train2017.merge(prop17, how='left', on=['parcelid'])
def calculate_features(df):

    df['N-zip_count'] = df['regionidzip'].map(zip_count)

    df['N-city_count'] = df['regionidcity'].map(city_count)

    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & \

                         (df['pooltypeid10']>0) & \

                         (df['airconditioningtypeid']!=5))*1 

    df['mean_area'] = df['regionidneighborhood'].map(meanarea)

    df['med_year'] = df['regionidneighborhood'].map(medyear)

    df['med_lat'] = df['regionidneighborhood'].map(medlat)

    df['med_long'] = df['regionidneighborhood'].map(medlong)

    df['taxrate'] = df['taxamount']/df['taxvaluedollarcnt']

    df['taxXcars'] = df['taxrate']*df['garagecarcnt']



def calculate_features17(df):

    df['N-zip_count'] = df['regionidzip'].map(zip_count17)

    df['N-city_count'] = df['regionidcity'].map(city_count17)

    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & \

                         (df['pooltypeid10']>0) & \

                         (df['airconditioningtypeid']!=5))*1 

    df['mean_area'] = df['regionidneighborhood'].map(meanarea17)

    df['med_year'] = df['regionidneighborhood'].map(medyear17)

    df['med_lat'] = df['regionidneighborhood'].map(medlat17)

    df['med_long'] = df['regionidneighborhood'].map(medlong17)

    df['taxrate'] = df['taxamount']/df['taxvaluedollarcnt']

    df['taxXcars'] = df['taxrate']*df['garagecarcnt']
calculate_features(train2016)

calculate_features17(train2017)
train = pd.concat([train2016, train2017], axis = 0)
train['month'] = pd.to_datetime(train['transactiondate']).dt.month

train['year'] = pd.to_datetime(train['transactiondate']).dt.year

train['yearmonth'] = 100*train.year+train.month

select_2016 = train['year']==2016

basedate = pd.to_datetime('2015-11-17').toordinal()

ordinal = pd.to_datetime(train.transactiondate).apply(lambda x: x.toordinal()-basedate)

train['cos_t'] = ( ordinal*(2*np.pi/365.25) ).apply(np.cos)

train['sin_t'] = ( ordinal*(2*np.pi/365.25) ).apply(np.sin)
train.columns
def impute_nas(train_df, test_df, feat):

    meds = train_df.median()

    for f in feat:

        imputed = meds[f]

        train_df[f] = train_df[f].replace(np.nan, meds[f])

        test_df[f] = test_df[f].replace(np.nan, meds[f])
static_features = [

    'finishedsquarefeet12',  # 678821

    'taxrate',               # 678811

    'garagetotalsqft',

    'garagecarcnt', 

    'N-zip_count', 

    'taxXcars'

]

features = static_features + ['cos_t', 

                              'sin_t']
data = train[select_2016]

n = np.sum(select_2016)

k = 5



X = data[features]

y = data.logerror

kf = cross_val.KFold(n, k=k)

avgmae = 0

for train_index, test_index in kf:

    X_train_, X_test_, y_train, y_test = cross_val.split(train_index, test_index, X, y)

    X_train = pd.DataFrame(X_train_.copy(), columns=features)

    X_test = pd.DataFrame(X_test_.copy(), columns=features)

    impute_nas(X_train, X_test, features)

    reg = QuantReg(y_train, sm.add_constant(X_train)).fit(q=.5) #,max_iter=2500)

    ypred = reg.predict(sm.add_constant(X_test,has_constant='add'))

    mae = mean_absolute_error(y_test, ypred)

    print( "Fold MAE: ", mae )

    avgmae += mae

avgmae /= k

print("\nFeatures:\n", features, "\n\nAverage MAE: ", avgmae)
# Test model on 2017 data



data = train[select_2016]

n = np.sum(select_2016)

test = train[~select_2016]



X = data[features].copy()

y = data.logerror

X_test = test[features].copy()

y_test = test.logerror



impute_nas(X, X_test, features)



reg = QuantReg(y, sm.add_constant(X)).fit(q=.5)

reg.summary()
ypred = reg.predict(sm.add_constant(X_test,has_constant='add'))



print( "Baseline MAE: ", mean_absolute_error(y_test, 0*ypred) )

print( "Model MAE:    ", mean_absolute_error(y_test, ypred) )
ypred.shape
test.parcelid.shape
simpLADpreds = pd.DataFrame({"ParcelId":test.parcelid, "pred":ypred})
simpLADpreds.to_csv('simpLADpreds17.csv',index=False)
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)



test_df = pd.merge( sample_submission[['ParcelId']], 

                    prop17.rename(columns = {'parcelid': 'ParcelId'}), 

                    how = 'left', on = 'ParcelId' )
# Train on full data set



data = train.copy()



calculate_features17(test_df)

impute_nas(data, test_df, static_features)



X = data[features]

y = data.logerror



reg = QuantReg(y, sm.add_constant(X)).fit(q=.5)

reg.summary()
y_preds = []

for tdate in ['2016-10-15', '2016-11-15', '2016-12-15']:

    test_df['transactiondate'] = tdate

    ordinal = pd.to_datetime(test_df.transactiondate).apply(lambda x: x.toordinal()-basedate)

    test_df['cos_t'] = ( ordinal*(2*np.pi/365.25) ).apply(np.cos)

    test_df['sin_t'] = ( ordinal*(2*np.pi/365.25) ).apply(np.sin)

    X_test = sm.add_constant(test_df[features],has_constant='add')

    pred = reg.predict(X_test)

    y_pred=[]

    for i,predict in enumerate(pred):

       y_pred.append(str(round(predict,5)))

    y_preds.append(np.array(y_pred))
output = pd.DataFrame({'ParcelId': sample_submission['ParcelId'].astype(np.int32),

       '201610': y_preds[0], '201611': y_preds[1], '201612': y_preds[2],

       '201710': y_preds[0], '201711': y_preds[1], '201712': y_preds[2]})



cols = output.columns.tolist()

cols
cols = cols[-1:] + cols[:-1]

output = output[cols]

output.head()
output.to_csv('simplad2017.csv', index=False)
features