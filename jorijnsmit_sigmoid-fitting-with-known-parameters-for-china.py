import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.optimize import curve_fit

from tqdm.notebook import tqdm
#%matplotlib notebook

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 8]



warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv',

    parse_dates=['Date']).drop(['Lat', 'Long'], axis=1)

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv',

    parse_dates=['Date']).drop(['Lat', 'Long'], axis=1)

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv',

    index_col='ForecastId')
# artifact(s) in the dataset:

# https://www.kaggle.com/marychin/check-non-cumulative-confirmedcases-fatalities

train.iloc[6425,4] = 0
def sigmoid(x, m, alpha, beta):

    """𝔼(𝑌𝑡) = 𝑀/(1 + exp(−𝛽(𝑡 − 𝛼)), with

    𝑌𝑡 the accumulated observed number (of deaths or cases) at day 𝑡 after a specific date,

    𝑀 the expected maximum number,

    𝛼 the number of days at which the expected number of counts is half way the maximum, and

    𝛽 > 0 the growth parameter.

    https://assets.tue.nl/fileadmin/content/pers/2020/03%20March/TUe%20-%20Technical_Report_Prediction_Corona_Virus.pdf

    """

    return m / ( 1 + np.exp(-beta * (x - alpha)))

  



def get_curve(covid, which):

    covid['DaysPassed'] = covid['Date'].dt.dayofyear

    curve = covid[covid[which] > 0].set_index('DaysPassed')[which]

    if curve.index.size > 3:

        return curve

    



def plot_curve(curve, test, name, plot_n, popt, ax):

    if curve is not None:

        _ = curve.plot(ax=ax[plot_n % 5, plot_n // 5], title=name)

        _.set_xlabel('')

        x = np.append(curve[:-12].index.values, test['Date'].dt.dayofyear.values)

        y = sigmoid(x, popt[0], popt[1], popt[2])

        pd.Series(y, x).plot(ax=ax[plot_n % 5, plot_n // 5], style=':')

    else:

        pd.Series(0).plot(ax=ax[plot_n % 5, plot_n // 5], title=name)



    

def predict_curve(covid, test, popt, which):

    train_curve = get_curve(covid, which)

    if train_curve is not None:

        x_train = train_curve.index.values

        y_train = train_curve.values

        popt, _ = curve_fit(sigmoid, x_train, y_train, p0=popt, maxfev=1000000)

        x_test = test['Date'].dt.dayofyear.values

        y_test = sigmoid(x_test, popt[0], popt[1], popt[2])

        test[which] = y_test

        return test.set_index('ForecastId')[which].astype('int'), popt

    return None, None





def append_predictions(train, test, popts):

    cases_popt, fatalities_popt = popts

    cases, cases_popt = predict_curve(train, test, cases_popt, 'ConfirmedCases')

    if cases is not None:

        CASES_ALL.append(cases)

    fatalities, fatalities_popt = predict_curve(train, test, fatalities_popt, 'Fatalities')

    if fatalities is not None:

        FATALITIES_ALL.append(fatalities)

    return cases_popt, fatalities_popt

   

    

def known_popt(country, region):

    known = {}

    known['cases'] = {

        'Anhui': [993, 13.3, 0.28],

        'Beijing': [411, 11.9, 0.22],

        'Chongqing': [573, 11.2, 0.24],

        'Fujian': [294, 10.7, 0.26],

        'Gansu': [96, 11.7, 0.26],

        'Guangdong': [1342, 11.7, 0.28],

        'Guangxi': [252, 12.2, 0.22],

        'Guizhou': [147, 14.9, 0.3],

        'Hainan': [170, 12.6, 0.23],

        'Hebei': [319, 14.9, 0.23],

        'Heilongjiang': [482, 15.7, 0.27],

        'Henan': [1270, 12.7, 0.28],

        'Hubei': [67625, 18.7, 0.24],

        'Hunan': [1018, 11.9, 0.29],

        'Inner Mongolia': [76, 13.5, 0.22],

        'Jiangsu': [635, 13.5, 0.26],

        'Jiangxi': [937, 13.1, 0.29],

        'Jilin': [92, 13.6, 0.34],

        'Liaoning': [122, 10.3, 0.26],

        'Ningxia': [74, 14.4, 0.21],

        'Qinghai': [18, 9.1, 0.38],

        'Shaanxi': [244, 11.6, 0.27],

        'Shandong': [781, 17.5, 0.15],

        'Shanghai': [336, 10.6, 0.28],

        'Shanxi': [133, 11.6, 0.31],

        'Sichuan': [535, 13.2, 0.21],

        'Tianjin': [137, 13.9, 0.21],

        'Xinjiang': [77, 15.6, 0.22],

        'Yunnan': [172, 10.1, 0.27],

        'Zhejiang': [1195, 10.5, 0.32],

        'China': [680, 13.3, 0.265],

    }

    known['fatalities'] = {

        'Heilonggjiang': [12.9, 18.6, 0.25],

        'Henan': [21.7, 22.5, 0.25],

        'Hubei': [3007, 23.6, 0.17],

    }

    if region in known['cases']:

        cases_popt = known['cases'][region]

    elif country in known['cases']:

        cases_popt = known['cases'][country]

    else:

        cases_popt = [5000, 100, 0.2]

        

    if region in known['fatalities']:

        fatalities_popt = known['fatalities'][region]

    elif country in known['fatalities']:

        fatalities_popt = known['fatalities'][country]

    else:

        fatalities_popt = [100, 150, 0.25]

    

    return cases_popt, fatalities_popt

    

    

def main():

    n = -1

    for country in tqdm(train['Country/Region'].unique()):

        country_train = train[train['Country/Region'] == country].copy()

        country_test = test[test['Country/Region'] == country].copy()

        if not country_train['Province/State'].isna().all():

            for region in country_train['Province/State'].unique():

                region_train = country_train[country_train['Province/State'] == region].copy()

                region_test = country_test[country_test['Province/State'] == region].copy()

                cases_popt, fatalities_popt = append_predictions(region_train, region_test, known_popt(country, region))

                if region in ['Hubei', 'New York', 'Hunan', 'California', 'France', 'Netherlands']:

                    n += 1

                    plot_curve(get_curve(region_train, 'ConfirmedCases'), region_test, region, n, cases_popt, AX)

                    plot_curve(get_curve(region_train, 'Fatalities'), region_test, region, n, fatalities_popt, AXX)

        else:

            cases_popt, fatalities_popt = append_predictions(country_train, country_test, known_popt(country, None))

            if country in ['Italy', 'Spain', 'Mexico', 'Mongolia']:

                n += 1

                plot_curve(get_curve(country_train, 'ConfirmedCases'), country_test, country, n, cases_popt, AX)

                plot_curve(get_curve(country_train, 'Fatalities'), country_test, country, n, fatalities_popt, AXX)
CASES_ALL = []

FIG, AX = plt.subplots(5, 2)

FIG.suptitle('Confirmed Cases')



FATALITIES_ALL = []

FIGG, AXX = plt.subplots(5, 2)

FIGG.suptitle('Fatalities')



main()



FIG.subplots_adjust(hspace=0.5)

FIGG.subplots_adjust(hspace=0.5)
final = pd.DataFrame(pd.concat(CASES_ALL).reindex(index=submission.index, fill_value=1))

final = final.join(pd.DataFrame(pd.concat(FATALITIES_ALL).reindex(index=submission.index, fill_value=0)))

final = final.where(final['Fatalities'] <= final['ConfirmedCases'], final['ConfirmedCases'] * 0.05, axis=0)

final.to_csv('submission.csv')