# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_training_csv(path):

    """helper function to regularize the preprocessing of dataframes"""

    df = pd.read_csv(path, header=0, parse_dates=['Date'])

    #df.drop(df[((df['ConfirmedCases'] == 0) & (df['Fatalities'] == 0))].index, inplace=True)

    df['ConfirmedCases_log1p'] = df['ConfirmedCases'].map(np.log1p)

    df['Fatalities_log1p'] = df['Fatalities'].map(np.log1p)

    df.drop(['Province/State', 'Country/Region','Lat','Long'], axis=1, inplace=True)

    return df



def load_case_data_from_csv(path):

    """load case count data where it's > 0 only and drop unneeded columns"""

    df = pd.read_csv(path, header=0, parse_dates=['Date'])

    df.drop(df[(df['ConfirmedCases'] == 0)].index, inplace=True)

    df['ConfirmedCases_log1p'] = df['ConfirmedCases'].map(np.log1p)

    df.drop(['Fatalities', 'Province/State', 'Country/Region','Lat','Long'], axis=1, inplace=True)

    return df



def load_fatality_data_from_csv(path):

    """load fatality count data where it's > 0 only and drop unneeded columns"""

    df = pd.read_csv(path, header=0, parse_dates=['Date'])

    df.drop(df[(df['Fatalities'] == 0)].index, inplace=True)

    df['Fatalities_log1p'] = df['Fatalities'].map(np.log1p)

    df.drop(['ConfirmedCases', 'Province/State', 'Country/Region','Lat','Long'], axis=1, inplace=True)

    return df



def rmsle(y_true, y_pred):

    """return the root mean squared logarithmic error: square root of the mean squared error of the natural log 

    of (value plus 1): sqrt(mean(power(log1p(p)-log1p(a),2)))

    """

    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred),2)))







from scipy.stats import linregress



full_df = load_training_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv") # for reference

cdf = load_case_data_from_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv") # just case counts

fdf = load_fatality_data_from_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv") # just fatality counts



y_cases_train = cdf['ConfirmedCases_log1p'].values

y_cases_true = cdf['ConfirmedCases'].values

y_fatalities_train = fdf['Fatalities_log1p'].values

y_fatalities_true = fdf['Fatalities'].values



# make some x values for regression purposes

case_xs = range(0, len(y_cases_train))

fat_xs = range(0, len(y_fatalities_train))



#

# linear regressionon against log1p transformed data

#



cslope, cintercept, cr, cp, csterr = linregress(case_xs, y_cases_train)

fslope, fintercept, fr, fp, fsterr = linregress(fat_xs, y_fatalities_train)

print("case log1p(y) = mx + b: {0:.4f}x + {1:.4f} sterr {2:.4f} r={3:.4f}".format(cslope, cintercept, csterr, cr))

print("fatality log1p(y) = mx + b:{0:.4f}x + {1:.4f} sterr {2:.4f} r={3:.4f}".format(fslope, fintercept, fsterr, fr))



#

# polyfit the transformed data

#



poly_degrees = 5

zcase = np.polyfit(case_xs, y_cases_train, deg=poly_degrees)

zfat = np.polyfit(fat_xs, y_fatalities_train, deg=poly_degrees)

case_poly = np.poly1d(zcase)

fat_poly = np.poly1d(zfat)



#

# check linear fit against existing data

#



y_cases_pred_linear = np.round(np.expm1(case_xs * cslope + cintercept), 0)

y_fatalities_pred_linear = np.round(np.expm1(fat_xs * fslope + fintercept), 0)

case_rmsle = rmsle(y_cases_true, y_cases_pred_linear)

fat_rmsle = rmsle(y_fatalities_true, y_fatalities_pred_linear)

plt.plot(case_xs, y_cases_true,'o')

plt.plot(case_xs, y_cases_pred_linear, '-', label="RMSLE={0:.4f}".format(case_rmsle))

plt.plot(fat_xs, y_fatalities_true, 'o')

plt.plot(fat_xs, y_fatalities_pred_linear, '-', label="RMSLE={0:.4f}".format(fat_rmsle))

plt.legend()

plt.show()



#

# check poly fit against existing data

#



y_cases_pred_poly = np.round(np.expm1(case_poly(case_xs)))

y_fatalities_pred_poly = np.round(np.expm1(fat_poly(fat_xs)))

case_poly_rmsle = rmsle(y_cases_true, y_cases_pred_poly)

fat_poly_rmsle = rmsle(y_fatalities_true, y_fatalities_pred_poly)

plt.plot(case_xs, y_cases_true, 'o')

plt.plot(case_xs, y_cases_pred_poly, '-', label="RMSLE={0:.4f}".format(case_poly_rmsle))

plt.plot(fat_xs, y_fatalities_true, 'o')

plt.plot(fat_xs, y_fatalities_pred_poly, '-', label='RMSLE={0:.4f}'.format(fat_poly_rmsle))

plt.legend()

plt.show()

                                      

#print(full_df.tail())



# load test data and predict based on above regression

test_df = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

print(test_df.head())

print(test_df.shape)

test_xs = range(2,test_df.shape[0]+2)

print(len(test_xs))

#print(list(test_xs))

test_df['xs'] = test_xs

#test_df['ConfirmedCases'] = test_df.apply(lambda x:np.round(np.expm1(x['xs']*cslope + cintercept)), axis=1)

#test_df['Fatalities'] = test_df.apply(lambda x: np.round(np.expm1(x['xs']*fslope + fintercept)), axis=1)

test_df['ConfirmedCases'] = test_df.apply(lambda x: np.round(np.expm1(case_poly(case_xs))), axis=1)

test_df['Fatalities'] = test_df.apply(lambda x: np.round(np.expm1(fat_poly(fat_xs))), axis=1)



print(test_df[test_df['Date'] > '2020-03-18'])



sub_df = test_df[['ForecastId','ConfirmedCases','Fatalities']]

sub_df.to_csv('submission.csv', index=False, header=True)