# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from scipy.stats import linregress, norm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def logistic(x, K, a, b):

    """a logistic curve generating function

    x: x values

    K: the carrying capacity; the maximum value the population can grow to.

    a: shape value 1; influences the length of the left tail

    b: shape value 2, must be < 0

    

    a and b together approximate r_max below

    

    The calculus version of this is:

    dN/dT = r_max * ((K-N)/K) * N

    Where r_max is the maximum per-capita rate of increase, K is the carrying capacity, and N is the number of individuals in the population.

    Taken from: https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth



    Code equation taken from: http://www.curve-fitting.com/aids.htm



    """

    xrun = np.array(x)

    krun = np.float128(K)

    arun = np.float128(a)

    brun = np.float128(b)

    #if b >= 0:

        #return None



    return K / (1 + np.exp(a + b * xrun))



def rmsle(y_true, y_pred):

    """return root mean squared log error between true and predicted value lists"""

    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred),2)))



train_df = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv', header=0, parse_dates=['Date'])

test_df = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv', header=0, parse_dates=['Date'])



# drop training dates on or after the first testing date to prevent data leakage

train_df = train_df.loc[train_df['Date'] < test_df['Date'].min()]



# do some feature engineering on the training data

train_df['Area'] = train_df['Country_Region'].str.cat(train_df['Province_State'], sep="/", na_rep='').str.replace('\/$', '')

train_df['ConfirmedCases_log1p'] = train_df.apply(lambda x: np.log1p(x['ConfirmedCases']), axis=1)

train_df['FatalityRatio'] = train_df.apply(lambda x:  x['Fatalities'] / x['ConfirmedCases'] if x['ConfirmedCases'] > 0 else np.nan, axis=1)



# add the area column to the test data too

test_df['Area'] = test_df['Country_Region'].str.cat(test_df['Province_State'], sep="/", na_rep='').str.replace('\/$', '')



#sub_df = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv', header=0)

print(train_df.shape)

print(train_df.columns)

print(test_df.shape)

print(test_df.columns)

#print(sub_df.shape)

#print(sub_df.columns)


# make a list of all the different areas

area_list = list(train_df['Area'].unique())



# create a submission dataframe

sub_df = pd.DataFrame({'ForecastId':[],

                      'ConfirmedCases': [],

                      'Fatalities': []

                      },dtype=np.int64)



# for each area in the unique list of areas

for one_area in area_list:



    # isolate one area's worth of data

    X_train = train_df.loc[train_df['Area'] == one_area]

    X_test = test_df.loc[test_df['Area'] == one_area]

    

    # get x and y values for modeling

    xs = range(0, X_train.shape[0])

    y_train_case = list(X_train['ConfirmedCases'])

    y_train_fat = list(X_train['Fatalities'])



    # prepare the range of x values needed for forecasting

    forecastxs = range(X_train.shape[0], X_train.shape[0] + X_test.shape[0])

    

    # decide if there's enough data to do logistic curve fitting, or fall back to linear regression

    if (len(X_train['ConfirmedCases'].unique()) > 4):

        

        # first guess as to the values needed in the logistic function, for curve_fit

        case_p0 = [1000000, 25, -.1] 

        # fit a logistic curve for case count with 10k iterations and initial values stored in p0

        case_opt, case_cov = curve_fit(logistic, xs, y_train_case, maxfev=500000, p0=case_p0)

    

        y_fitted_train_case = np.round(logistic(xs, case_opt[0], case_opt[1], case_opt[2]), 0)



        # forecast the values for case count from the curve we just fit

        y_pred_case = np.round(logistic(forecastxs, case_opt[0], case_opt[1], case_opt[2]), 0)    

    

        # calculate the value of 1 std dev for each of those measures

        case_sd = np.sqrt(np.diag(case_cov))



        # plot all lines for context

        low_y_pred_case = np.round(logistic(forecastxs, case_opt[0]-case_sd[0], case_opt[1]-case_sd[1], case_opt[2]-case_sd[2]))

        high_y_pred_case = np.round(logistic(forecastxs, case_opt[0]+case_sd[0], case_opt[1]+case_sd[1], case_opt[2]+case_sd[2]))

             

    else:

    

        # perform linear regression

        m, b, r, p, std_err = linregress(xs, y_train_case)

        y_fitted_train_case = np.maximum(np.zeros(len(xs)), np.round((m * xs) + b, 0))

        y_pred_case = np.round((m * forecastxs) + b, 0)

    

    # model fatalities

    

    fatality_ratio = np.mean(list(X_train.loc[np.isnan(X_train['FatalityRatio']) == False, 'FatalityRatio']))

    if np.isnan(fatality_ratio) == True:

        

        # use the global average for this country

        fatality_ratio = X_train['FatalityRatio'].mean()

        

    y_fat_ratio_train = np.round(X_train['ConfirmedCases'] * fatality_ratio, 0)

    y_fat_ratio_forecast = np.round(fatality_ratio * y_pred_case, 0)

    #plt.plot(xs, X_train['Fatalities'], '-', label='training fatality ratio')

    #plt.plot(xs, y_fat_ratio_train, '.', label='fitted fatality ratio')

    #plt.plot(forecastxs, y_fat_ratio_forecast, 'o', label='forecast fatalities')

    #plt.legend(loc='best')

    #plt.plot(ndx, normal_pdf, 'o', label='normal pdf')

    #plt.title(one_area)

    #plt.show()

    

    # write out this information to the submission dataframe

    

    ids = test_df.loc[test_df['Area'] == one_area, 'ForecastId']

    sub_df = pd.concat([sub_df, pd.DataFrame({'ForecastId' : ids,

                                            'ConfirmedCases' : y_pred_case,

                                            'Fatalities' : y_fat_ratio_forecast

                                            },dtype=np.int64)])

    

    train_case_rmsle = rmsle(y_train_case, y_fitted_train_case)

    print("{0} rmsle cases: {1:.3f}".format(one_area, train_case_rmsle))

    #plt.plot(xs, y_train_case, '-', label='training')

    #plt.plot(xs, y_fitted_train_case, '.', label='fitted training')

    #plt.plot(forecastxs, y_pred_case, 'x', label='forecaset')

    #plt.legend(loc='best')

    #plt.title(one_area)

    #plt.show()

        
print(sub_df.describe())

print(sub_df.isnull().sum())



#

# there are countries that have not reported any cases, so they have no fatalities  do an NA fill

#



sub_df['Fatalities'] = sub_df['Fatalities'].fillna(value=0)

sub_df['Fatalities'] = sub_df['Fatalities'].astype('int64')

print(sub_df.describe())

print(sub_df.isnull().sum())

print(sub_df.info())

print(sub_df.shape)
# write out the header, then commit and submit!

sub_df.to_csv('submission.csv', header=True, index=False)

print("Complete.")