# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # somewhat fancier plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def rmsle(y_true, y_pred):

    """return root mean squared log error between true and predicted value lists"""

    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred),2)))



# load the data and do some preliminary column engineering



train_df = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv", header=0, parse_dates=['Date'])

test_df = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv", header=0, parse_dates=['Date'])

hs_df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv", header=0)



hs_numeric_cols = ['Health_exp_pct_GDP_2016', 'Health_exp_public_pct_2016',

       'Health_exp_out_of_pocket_pct_2016', 'Health_exp_per_capita_USD_2016',

       'per_capita_exp_PPP_2016', 'External_health_exp_pct_2016',

       'Physicians_per_1000_2009-18', 'Nurse_midwife_per_1000_2009-18',

       'Specialist_surgical_per_1000_2008-18',

       'Completeness_of_birth_reg_2009-18',

       'Completeness_of_death_reg_2008-16']



# make a singluar area that will be country and subdivision together



train_df['Area'] = train_df['Country_Region'].str.cat(train_df['Province_State'], sep="/", na_rep='').str.replace('\/$', '')

test_df['Area'] = test_df['Country_Region'].str.cat(test_df['Province_State'], sep="/", na_rep='').str.replace('\/$', '')

hs_df['Area'] = hs_df['Country_Region'].str.cat(hs_df['Province_State'], sep='/', na_rep='').str.replace('\/$', '')

hs_df.set_index('Area', inplace=True)

train_df.set_index('Area', inplace=True)



# drop training dates on or after the first testing date to prevent data leakage

min_test_date = test_df['Date'].min()

train_df = train_df.loc[train_df['Date'] < min_test_date]



print("{0} unique dates in training".format(len(list(train_df['Date'].unique()))))

print("{0} unique dates in testing".format(len(list(test_df['Date'].unique()))))



train_df = train_df.join(hs_df, on='Area', rsuffix='_hs')



# calculate the case:fatality ratio

train_df['FatalityRatio'] = train_df.apply(lambda x: x['Fatalities'] / x['ConfirmedCases'] if x['ConfirmedCases'] > 0 else np.nan, axis=1)



# fill in NaN values in the health system columns of the test and train set with the median value calculated in the test set

for c in hs_numeric_cols:

    median = np.nanmedian(train_df[c])

    train_df[c].fillna(value=median, axis=0, inplace=True)



print(train_df[hs_numeric_cols].isnull().sum())

test_df['ConfirmedCases'] = pd.Series()

test_df['Fatalities'] = pd.Series()

print(test_df.columns)

from scipy.stats import linregress



polyfit_degree = 2



fitted_lin_slopes = []

fatality_slopes = []

fatality_ratios = []



# to build prediction

predicted_ConfirmedCases = []

predicted_Fatalities = []



docs = []

nurses = []

pub_spend = []

per_cap_ppp = []



train_df['Area'] = train_df.index

#print(train_df['Area'])



#area_list = list(train_df.index.unique())

area_list = list(train_df['Area'].unique())



# get the number of days we need to predict

days_to_predict = len(list(test_df['Date'].unique()))

days_of_data = len(list(train_df['Date'].unique()))



sub_df = pd.DataFrame({'ForecastId':[],

                      'ConfirmedCases': [],

                      'Fatalities': []

                      },dtype=np.int64)



print("will predict {0} days from {1} days of data".format(days_to_predict, days_of_data))



for one_area in area_list:

    

    # build model off of days since 100th case, as this is more stable

    area_df = train_df.loc[(train_df['Area'] == one_area) & (train_df['ConfirmedCases'] >= 100)]

    

    # if that isn't possible, use everything

    if area_df.shape[0] < 3:

    

        area_df = train_df.loc[train_df['Area'] == one_area]

    

    # log transform the cases, model growth as log linear, predict, and assess accuracy

    area_df['ConfirmedCases_log1p'] = area_df.apply(lambda x: np.log1p(x['ConfirmedCases']), axis=1)

    ys = list(area_df['ConfirmedCases_log1p'])

    xs = range(1, area_df.shape[0] + 1)

    logm, logb, logr, logp, logstd_err = linregress(xs, ys)

    y_case_pred_lin = np.maximum(np.zeros(len(xs)), np.round(np.expm1((xs * logm) + logb), 0))

    case_logrmsle = rmsle(list(area_df['ConfirmedCases']), y_case_pred_lin)

    print("{0} case loglin log1p(y) = {1:.3f}x + {2:.3f} r={3:.3f} sterr={4:.3f} rmsle={5:.3f}".format(one_area, logm, logb, logr, logstd_err, case_logrmsle))



    # model fatalities as mean case:fatality ratio for this area

    ratios = list(area_df['FatalityRatio'])

    mean_fatality_ratio = np.nanmean(ratios)

    

    if np.isnan(mean_fatality_ratio):

        

        # make global mean fatality ratio

        print("-->{0} had no fatalities, going with global mean of {1:.3f}".format(one_area, np.nanmean(list(train_df['FatalityRatio']))))

        mean_fatality_ratio = np.nanmean(list(train_df['FatalityRatio']))

    

    xs = list(area_df['ConfirmedCases'])

    y_fat_pred = np.round(np.array(xs) * mean_fatality_ratio, 0)

    fat_rmsle = rmsle(list(area_df['Fatalities']), y_fat_pred)

    print("{0} mean fatality ratio fat:cases={1:.3f} rmsle={2:.3f}".format(one_area, mean_fatality_ratio, fat_rmsle))



    # model fatalities as regressed from fatality data

    ys = list(area_df['Fatalities'])

    fm, fb, fr, fp, fstd_err = linregress(xs, ys)

    y_fat_lin_pred = np.maximum(np.zeros(len(xs)), np.round((np.array(xs) * fm) + fb, 0))

    fat_lin_rmsle = rmsle(list(area_df['Fatalities']), y_fat_lin_pred)

    print("{0} fatality lin reg y = {1:.3f}x + {2:.3f} r={3:.3f} sterr={4:.3f} rmsle={5:.3f}".format(one_area, fm, fb, fr, fstd_err, fat_lin_rmsle))

   

    # predict the ConfirmedCases for this area

    pred_xs = range(area_df.shape[0], area_df.shape[0] + days_to_predict)

    predicted_ConfirmedCases = np.maximum(np.zeros(len(pred_xs)), np.round(np.expm1((pred_xs * logm) + logb), 0))

    print(predicted_ConfirmedCases)

    # predict fatalities from the better of the two metrics for this area

    

    if fat_lin_rmsle < fat_rmsle:

        

        # use linear regression of raw data

        predicted_Fatalities = np.maximum(np.zeros(len(pred_xs)), np.round((np.array(pred_xs) * fm) + fb, 0))

    

    else:

        

        # use case:fatality ratio

        predicted_Fatalities = np.round(np.array(predicted_ConfirmedCases) * mean_fatality_ratio, 0)



    # add data to submission dataframe

    ids = test_df.loc[test_df['Area'] == one_area, 'ForecastId']

    sub_df = pd.concat([sub_df, pd.DataFrame({'ForecastId' : ids,

                                            'ConfirmedCases' : predicted_ConfirmedCases,

                                            'Fatalities' : predicted_Fatalities

                                            },dtype=np.int64)])

    

    # catalog data points for eventual modeling

    docs.append(train_df.loc[(train_df['Area'] == one_area),'Physicians_per_1000_2009-18'].unique()[0])

    nurses.append(train_df.loc[(train_df['Area'] == one_area),'Nurse_midwife_per_1000_2009-18'].unique()[0])

    pub_spend.append(train_df.loc[(train_df['Area'] == one_area),'Health_exp_public_pct_2016'].unique()[0])

    per_cap_ppp.append(train_df.loc[(train_df['Area'] == one_area),'per_capita_exp_PPP_2016'].unique()[0])

    fitted_lin_slopes.append(logm)

    fatality_ratios.append(mean_fatality_ratio)

    fatality_slopes.append(fm)



print("Submission dataframe:")

print(sub_df.describe())



modeling_df = pd.DataFrame({'Area' : area_list, 

                            'ConfirmedCases_fitted_lin_slope' : fitted_lin_slopes, 

                            'Fatalities_slope' : fatality_slopes,

                            'Fatality_ratio' : fatality_ratios, 

                            'Doctor_per_1k' : docs,

                            'Nurse_midwife_per_1k' : nurses, 

                            'Pct_health_spending_public_money' : pub_spend, 

                            'Per_capita_health_spend_PPP' : per_cap_ppp

                           })
# write it out

print(sub_df.dtypes)

print(sub_df.head())

print(sub_df.describe())

sub_df.to_csv('submission.csv', header=True, index=False)

#print(modeling_df['Doctor_per_1k'])

print(modeling_df.describe())

print(modeling_df.isnull().sum())

print(sns.pairplot(modeling_df))

# We've got 28 null values in the fatality slope column; let's fill them in with the mean column value.



mean_fat_slope = np.nanmean(modeling_df['Fatalities_slope'])

modeling_df['Fatalities_slope'].fillna(mean_fat_slope, inplace=True)

print(modeling_df.isnull().sum())

                            
# split out our target values and drop them plus the row label Area



y_case_slope = modeling_df['ConfirmedCases_fitted_lin_slope']

y_fat_slope = modeling_df['Fatalities_slope']

y_fat_ratio = modeling_df['Fatality_ratio']

X = modeling_df.drop(['Area','ConfirmedCases_fitted_lin_slope', 'Fatalities_slope','Fatality_ratio'], axis=1)

print(X.columns)

print(X.shape)

# define a seed for repeatability

rand_seed = 112358



from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error



kf = KFold(n_splits=5, shuffle=True, random_state=rand_seed)

errors = []



target_dict = {'case_slope': y_case_slope,

              'fat_slope' : y_fat_slope,

              'fat_ratio' : y_fat_ratio

              }



# for each of the targets

for name, y in target_dict.items():

    

    # do cross-val splitting

    for train_index, test_index in kf.split(X):



        X_train, X_test, y_train, y_test = train_test_split(X.loc[train_index], y.loc[train_index], test_size=0.25, random_state=rand_seed)

        #X_train, X_test = np.array(X.loc[train_index]), np.array(X.loc[test_index])

        #y_train, y_test = np.array(y.loc[train_index]), np.array(y.loc[test_index])

        

        rfr = RandomForestRegressor()

        rfr.fit(X_train, y_train)

        print(rfr.estimators_[0])

        

        y_pred = rfr.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print("{0} mse: {1:.3f}".format(name, mean_squared_error(y_test, y_pred)))

        print("{0} mae: {1:.3f}".format(name, mean_absolute_error(y_test, y_pred)))

        print("{0} rmse: {1:.3f}".format(name, rmse))

        print("{0} score: {1:3f}".format(name, rfr.score(X_test, y_test)))

        

        errors.append(mean_absolute_error(y_test, y_pred))

        

        print("Feature importances: ")

        impdict = dict(list(zip(X.columns, rfr.feature_importances_)))

        print(impdict)

        

    print("{0} average mae: {1:.3f}".format(name, np.mean(errors)))