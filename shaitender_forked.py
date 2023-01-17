# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'



import matplotlib.pyplot as plt

%matplotlib inline 

!pip install mpld3

import mpld3

mpld3.enable_notebook()



from sklearn.metrics import mean_squared_log_error



from tqdm.notebook import tqdm
continents = pd.read_csv("/kaggle/input/data-for-covid-prediction/Countries-Continents.csv")

beds = pd.read_csv("/kaggle/input/data-for-covid-prediction/icu_beds.csv", header=0)

agegroups = pd.read_csv("/kaggle/input/data-for-covid-prediction/population.csv")

delta_alpha = pd.read_csv("/kaggle/input/data-for-covid-prediction/delta_alpha.csv")

fixed_vars = pd.read_csv("/kaggle/input/data-for-covid-prediction/fixed_vars.csv")

pop_info = pd.read_csv('/kaggle/input/data-for-covid-prediction/population_data.csv')



# competition data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', parse_dates=['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', parse_dates=['Date'])

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv', index_col=['ForecastId'])
DATE_BORDER = '2020-04-08'



# Create lookup dicts etc.

# 1. population

country_pop = pop_info.query('Type == "Country/Region"')

province_pop = pop_info.query('Type == "Province/State"')

population_country_lookup = dict(zip(country_pop['Name'], country_pop['Population']))

population_province_lookup = dict(zip(province_pop['Name'], province_pop['Population']))

population_province_lookup["Northwest Territories"] = 44800

population_province_lookup["Yukon"] = 35874

population_province_lookup["Anguilla"] = 15094

population_province_lookup["British Virgin Islands"] = 31196

population_province_lookup["Turks and Caicos Islands"] = 35446

# 2. continents

continent_lookup = dict(zip(continents["Country"], continents["Continent"]))

continent_lookup["Burma"] = "Asia"

continent_lookup["Kosovo"] = "Europe"

continent_lookup["MS Zaandam"] = "North America"

continent_lookup["West Bank and Gaza"] = "Asia"

continent_lookup["Fiji"] = "Asia"

continent_lookup["Papua New Guinea"] = "Asia"

# 3. beds

beds_lookup = dict(zip(beds["Country"], beds["ICU_Beds"]))

beds_lookup["Israel"] = 10.0

# 4. agegroups

agegroup_lookup = dict(zip(agegroups['Location'], agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89', '90_100']].values))

agegroup_lookup["Burma"] = agegroup_lookup["Myanmar"]

agegroup_lookup["Dominican Republic"] = agegroup_lookup["Dominica"]

agegroup_lookup["MS Zaandam"] = agegroup_lookup["Diamond Princess"]

agegroup_lookup["Holy See"] = np.asarray(np.ones(10))

agegroup_lookup["West Bank and Gaza"] = agegroup_lookup["Eastern Africa"] * (4685000 / sum(agegroup_lookup["Eastern Africa"]))





# 5. delta and alpha values

delta_1 = [float(x.replace(",", ".")) for x in list(delta_alpha.delta_1.values)]

delta_2 = [float(x.replace(",", ".")) for x in list(delta_alpha.delta_2.values)]

alpha_1 = [float(x.replace(",", ".")) for x in list(delta_alpha.alpha_1.values)]

alpha_2 = [float(x.replace(",", ".")) for x in list(delta_alpha.alpha_2.values)]

# 6. already fixed variables



fixed_vars_lookup = dict(zip(fixed_vars["Area"], [dict(zip(["shift", "R_0_start", "R_0_end", "k", "x0", "get_from"], vals)) for vals in fixed_vars[["shift", "R_0_start", "R_0_end", "k", "x0", "get_from"]].values]))

#!!!#

fixed_vars_lookup = {}

#!!!#



train['Province_State'] = train['Province_State'].replace('Georgia', 'Georgia (State)')

test['Province_State'] = test['Province_State'].replace('Georgia', 'Georgia (State)')

population_province_lookup['Georgia (State)'] = population_province_lookup['Georgia']



train['Area'] = train['Province_State'].fillna(train['Country_Region'])

test['Area'] = test['Province_State'].fillna(test['Country_Region'])





# https://www.kaggle.com/c/covid19-global-forecasting-week-1/discussion/139172

train['ConfirmedCases'] = train.groupby('Area')['ConfirmedCases'].cummax()

train['Fatalities'] = train.groupby('Area')['Fatalities'].cummax()



# Remove the leaking data

train_full = train.copy()

valid = train[train['Date'] >= test['Date'].min()]

train = train[train['Date'] < test['Date'].min()]



# Split the test into public & private

test['ConfirmedCases'] = 0

test['Fatalities'] = 0

test_public = test[test['Date'] <= DATE_BORDER]

test_private = test[test['Date'] > DATE_BORDER]



submission['ConfirmedCases'] = 0

submission['Fatalities'] = 0



FIRST_PRED_DATE = np.datetime64(test_public["Date"].min())

LAST_PRED_DATE = np.datetime64(test_private["Date"].max())

NUMBER_OF_PRED_DATES = np.timedelta64(LAST_PRED_DATE - FIRST_PRED_DATE, 'D').astype(int) + 1

NUMBER_OF_PUBLIC_PRED_DATES = len(test_public["Date"].unique())
from scipy.integrate import odeint

!pip install lmfit

from lmfit import Model
def extended_deriv_SEIR(y, t, beta, gamma, sigma, N, S_1, S_2, B):

    '''

    gamma, sigma, N, S_1, S_2: fixed floats

    beta, B: callable

    '''

    S, E, I, C, R, D = y



    dSdt = -beta(t) * I * S / N

    dEdt = beta(t) * I * S / N - sigma * E

    dIdt = sigma * E - 1/12.0 * S_1 * I - gamma * (1 - S_1) * I

    dCdt = 1/12.0 * S_1 * I - 1/7.5 * S_2 * min(B(t), C) - max(0, C-B(t)) - (1 - S_2) * 1/6.5 * min(B(t), C)

    dRdt = gamma * (1 - S_1) * I + (1 - S_2) * 1/6.5 * min(B(t), C)

    dDdt = 1/7.5 * S_2 * min(B(t), C) + max(0, C-B(t))

    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt
gamma = 1.0/9.0

sigma = 1.0/3.0



def logistic_R_0(t, R_0_start, k, x0, R_0_end):

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end



def SEIR_Model(days, agegroups, beds_per_100k, delta_agegroups, alpha_agegroups, **R0_kwargs):

    R_0_start, k, x0, R_0_end = 2.7, 0.2, 95, 1.4

    def beta(t):

        return logistic_R_0(t, **R0_kwargs) * gamma



    N = sum(agegroups)

    rel_freq_delta_agegroups = [agegroups[i]/N * delta_agegroups[i] for i in range(len(agegroups))]

    S_1 = sum(rel_freq_delta_agegroups)

    S_2 = sum([rel_freq_delta_agegroups[j] / S_1 * alpha_agegroups[j] for j in range(len(agegroups))])

    def B(t):

        # return beds_per_100k / 100_000 * N

        return 0.003*(beds_per_100k / 100_000 * N)*t + (beds_per_100k / 100_000 * N)



    y0 = N-1.0, 1.0, 0.0, 0.0, 0.0, 0.0

    t = np.linspace(0, days, days)

    ret = odeint(extended_deriv_SEIR, y0, t, args=(beta, gamma, sigma, N, S_1, S_2, B))

    S, E, I, C, R, D = ret.T

    R_0 = [beta(i)/gamma for i in range(len(t))]



    return t, S, E, I, C, R, D, R_0, B, S_1, S_2
def fit_extended_SEIR(y_data, agegroups, beds_per_100k, delta_agegroups, alpha_agegroups, fit_method="leastsq", outbreak_shift=45, fixed=[], **R_0_kwargs_minmax):



    max_days = outbreak_shift + len(y_data)

    if outbreak_shift >= 0:

        y_data = np.concatenate((np.zeros(outbreak_shift), y_data))

    else:

        y_data = y_data[-outbreak_shift:]



    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)  # x_data is just [0, 1, ..., max_days] array

    

    def SEIR_deaths(x, **R_0_kwargs):

        ret =  SEIR_Model(max_days, agegroups, beds_per_100k, delta_agegroups, alpha_agegroups, **R_0_kwargs)

        return ret[6][x]



    mod = Model(SEIR_deaths)



    for kwarg, (init, mini, maxi) in R_0_kwargs_minmax.items():

        mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)



    params = mod.make_params()

    for par in fixed:

        params[par].set(vary=False)



    result = mod.fit(y_data, params, method=fit_method, x=x_data)  # , fit_kws={'maxfev': 100}



    R_0_result_params = {}

    for val in R_0_kwargs_minmax:

        R_0_result_params[val] = result.params[val].value



    return result, R_0_result_params
def area_to_pred(area):

    print("Area: {}".format(area), end="    ")

    # 1.

    y_data_fat = train_full[train_full["Area"] == area]["Fatalities"].values

    y_data_cc = train_full[train_full["Area"] == area]["ConfirmedCases"].values

    country = train_full[train_full["Area"] == area]["Country_Region"].values[0]

    continent = continent_lookup[country]

    assert(y_data_fat.shape == train_full.Date.unique().shape == y_data_cc.shape)

    # 2.

    last_10_days = y_data_fat[-10:]

    if max(last_10_days) - min(last_10_days) < 10 or max(y_data_fat) < 10:

        cases_pred, fatalities_pred = simple_predict(y_data_fat, y_data_cc)

        assert(cases_pred.shape == (NUMBER_OF_PRED_DATES,) == fatalities_pred.shape)

    else:

        # 3.

        # beds

        beds = beds_lookup.get(area, beds_lookup.get(country, beds_lookup.get(continent, None)))

        assert(beds is not None)

        # agegroups and total pop.

        agegroups = get_agegroups(area, country, continent)

        N = sum(agegroups)

        assert(not np.isnan(N))

        # 4.

        fixed_vars = fixed_vars_lookup.get(area, fixed_vars_lookup.get(country, None))

        fixed_vars, fixed, shifts = handle_fixed_vars(fixed_vars)

        shifts = [int(i) for i in shifts]

        # 5.

        #if max(y_data_fat) < 100:

         #   fixed_vars = {"R_0_start": (3.5, 2.0, 5.0),

          #                "R_0_end": (0.9, 0.3, 3.5),

           #               "k": (3.2, 0.01, 5.0),

            #              "x0": (60, 0, 120)}

           # fixed = ["R_0_start"]

        

        new_shift, new_vars, cases_pred, fatalities_pred = predict_fatalities(y_data_cc, y_data_fat, beds, 

                                                                              agegroups, N, 

                                                                              fixed_vars, fixed, shifts) 

        cases_pred, fatalities_pred = cases_pred[new_shift:], fatalities_pred[new_shift:]

        # preds go from first data date until last day of private prediction PLUS 20 days extra

        # 6.

        # both 127,

        fatalities_pred = fatalities_pred[:-20]

        fatalities_pred = fatalities_pred[-NUMBER_OF_PRED_DATES:]

        try:

            cases_pred = extrapolate_cases(cases_pred, y_data_cc)

            cases_pred = cases_pred[-NUMBER_OF_PRED_DATES:]

        except:

            cases_pred, fatalities_pred = simple_predict(y_data_fat, y_data_cc)

        

        # preds go from first day of public pred until last day of private pred v/



        # 9.

       # if not isinstance(fixed_vars["get_from"], str):

        #    new_row = [area, new_shift, new_vars["R_0_start"], new_vars["R_0_end"], new_vars["k"], new_vars["x0"], np.nan]

         #   # either replace or add new row at area

          #  if area in fixed_vars_lookup:

           #     fixed_vars[fixed_vars["Area"]== area] = new_row

            #else:

             #   fixed_vars.loc[len(fixed_vars)] = new_row



    # 7.

    RMSLE_cc = np.sqrt(mean_squared_log_error(y_data_cc[-NUMBER_OF_PUBLIC_PRED_DATES:], 

                                                cases_pred[:NUMBER_OF_PUBLIC_PRED_DATES]))

    RMSLE_fa = np.sqrt(mean_squared_log_error(y_data_fat[-NUMBER_OF_PUBLIC_PRED_DATES:],

                                                fatalities_pred[:NUMBER_OF_PUBLIC_PRED_DATES]))

    print("RMSLE Confirmed Cases: {}, RMSLE Fatalities: {}".format(np.mean(RMSLE_cc), np.mean(RMSLE_fa)))

    # 8.

    test_public["Fatalities"][test_public["Area"] == area] = fatalities_pred[:NUMBER_OF_PUBLIC_PRED_DATES]

    test_public["ConfirmedCases"][test_public["Area"] == area] = cases_pred[:NUMBER_OF_PUBLIC_PRED_DATES]

    test_private["Fatalities"][test_private["Area"] == area] = fatalities_pred[NUMBER_OF_PUBLIC_PRED_DATES:]

    test_private["ConfirmedCases"][test_private["Area"] == area] = cases_pred[NUMBER_OF_PUBLIC_PRED_DATES:]
def extrapolate_cases(y_hat, y):



    def optfloat(intcoef, ys):

        xs = np.linspace(0, len(ys), len(ys), dtype=int)

        from scipy.optimize import curve_fit



        def f(t, m, b, c, d):

            t = t.astype(int)

            # return (m*t + b) * y_hat_cc[t + intcoef]

            return logistic_R_0(t, m, b, c, d) * y_hat[t + intcoef]

        

        popt, pcov = curve_fit(f, xs, ys)

        errsqr = np.linalg.norm(f(xs, *popt) - ys)

        return dict(errsqr=errsqr, floatcoef=popt)



    def errfun(intcoef, *args):

        ys = args[0]

        return optfloat(intcoef, ys)['errsqr']

    

    from scipy.optimize import brute

    ys = y

    grid = [slice(0, 20, 1)]

    intcoef = int(brute(errfun, grid, args=(ys,), finish=None))

    floatcoef = optfloat(intcoef, ys)["floatcoef"]

    m, b, c, d = floatcoef

    if intcoef > 0:

        y_fit = np.asarray([logistic_R_0(t, m, b, c, d) * y_hat[t + intcoef] for t in range(len(y_hat[:-intcoef]))])

    else:

        y_fit = np.asarray([logistic_R_0(t, m, b, c, d) * y_hat[t + intcoef] for t in range(len(y_hat))])

    lost_days = intcoef

    # print(intcoef, floatcoef)

    if 20 - lost_days > 0:

        y_fit = y_fit[:-(20-lost_days)]

    return np.asarray([max(0.0, x) for x in y_fit])
def predict_fatalities(y_cases, y_data_fat, beds, agegroups, N, fixed_vars, fixed, shifts):

    curr_ret = None

    curr_redchi = float("+inf")

    curr_shift = 0

    for shift in shifts:

        ret = fit_extended_SEIR(y_data=y_data_fat, agegroups=agegroups, beds_per_100k=beds, 

                                delta_agegroups=delta_1, alpha_agegroups=alpha_1, fit_method="least_squares", 

                                outbreak_shift=shift, fixed=fixed,

                                **fixed_vars)

        try:

            if ret[0].redchi < curr_redchi:

                curr_redchi = ret[0].redchi

                curr_ret = ret

                curr_shift = shift

        except:

            print("shift {}: no redchi".format(shift))

            continue

    assert(curr_ret is not None)

    start_date = np.datetime64(train_full["Date"].min()) - np.timedelta64(curr_shift,'D')

    x_ticks = np.arange(start_date, LAST_PRED_DATE + np.timedelta64(21, 'D'), step=np.timedelta64(1,'D')) # +20

    x_ticks = [pd.to_datetime(str(t)).strftime("%m/%d") for t in x_ticks]

    

    if curr_ret is None:

        print("Fitting unsuccessful.")

        cases_pred, fatalities_pred = simple_predict(y_data_fat, y_cases)

        return curr_shift, curr_ret[1], cases_pred, fatalities_pred

        

    

    t, S, E, I, C, R, D, R_0, B, S_1, S_2 = SEIR_Model(days=len(x_ticks), agegroups=agegroups, 

                                                       beds_per_100k=beds, delta_agegroups=delta_1, 

                                                       alpha_agegroups=alpha_1, 

                                                       **curr_ret[1])

    # cases_pred, fatalities_pred = E[-NUMBER_OF_PRED_DAYS:], D[-NUMBER_OF_PRED_DAYS:]

    return curr_shift, curr_ret[1], I, D
def simple_predict(y_data_fat, y_data_cc):

    ''' predict last value for all future days '''

    # print(y_data_fat[-1])

    return np.full((NUMBER_OF_PRED_DATES,), y_data_cc[-1]), np.full((NUMBER_OF_PRED_DATES,), y_data_fat[-1])
def handle_fixed_vars(fixed_vars):

    default_fixed_vars = {"R_0_start": (3.5, 2.0, 5.0),

                          "R_0_end": (0.9, 0.3, 3.5),

                          "k": (3.2, 0.01, 5.0),

                          "x0": (60, 0, 120)}

    default_shifts = [50, 30, 10, -10, -30, -50]

    if fixed_vars is None:

        return default_fixed_vars, [], default_shifts

    if isinstance(fixed_vars["get_from"], str):

        fixed_vars = fixed_vars_lookup[fixed_vars["get_from"]]

    ret = default_fixed_vars.copy()

    ret_fixed = []

    for var in ret.keys():

        if not np.isnan(fixed_vars[var]):

            ret[var] = (fixed_vars[var], float("-inf"), float("+inf"))

            ret_fixed.append(var)

    if not np.isnan(fixed_vars["shift"]):

        return ret, ret_fixed, [fixed_vars["shift"]]

    else:

        return ret, ret_fixed, default_shifts
def get_agegroups(area, country, continent):

    if area in agegroup_lookup:

        return agegroup_lookup[area]

    else:

        area_total_pop = population_country_lookup.get(area, population_province_lookup.get(area, None))

        region_agegroups = agegroup_lookup.get(country, agegroup_lookup.get(continent, None))

        assert(area_total_pop is not None and region_agegroups is not None)

        region_total_pop = sum(region_agegroups)

        return region_agegroups * (area_total_pop / region_total_pop)
all_areas = train_full.Area.unique().tolist()

failed_areas = set()
for x in tqdm(all_areas):

    try:

        area_to_pred(x)

    except:

        print("Area {} failed".format(x))

        failed_areas.add(x)
sub = submission.copy()

t_pub, t_priv = test_public.copy(), test_private.copy()



t_pub = t_pub.set_index('ForecastId')

t_priv = t_priv.set_index('ForecastId')

#sub = sub.set_index('ForecastId')



sub.update(t_pub)

sub.update(t_priv)

sub.reset_index(inplace=True)
sub.to_csv(r'submission.csv', index=False)

from IPython.display import FileLink

FileLink(r'submission.csv')