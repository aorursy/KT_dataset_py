import numpy as np

import pandas as pd

from statsmodels.tsa.arima_model import ARIMA

#import pmdarima as pm

import matplotlib.pyplot as plt

import matplotlib as mpl

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")



mpl.rcParams['axes.grid']=True

plt.rcParams.update({'figure.figsize':(8,5), 'figure.dpi':120})



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')



train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)

def fill_province(row):

    if pd.isna(row['Province_State']):

        row['Province_State'] = '_PROVINCE_' + row['Country_Region']

    return row



train = train.apply(fill_province, axis = 1)

test = test.apply(fill_province, axis = 1)
print('train starts:',train.Date.min())

print('train ends:', train.Date.max())

print('test starts:', test.Date.min())

print('test ends:', test.Date.max())
def get_country(df, country, province):

    country_df = df[(df['Country_Region'] == country) & (df['Province_State'] == province)]

    country_df = country_df.set_index(keys = 'Date')

    return country_df

def rmsle(y, y_hat):

    y_hat = y_hat.clip(0)

    res = np.sqrt(mean_squared_log_error(y, y_hat))

    return res



def evaluate_arima_model(X, arima_order):

    train_size = int(len(X) * 0.8)

    train, val = X[0:train_size], X[train_size:]

    history = [x for x in train]

    predictions = list()

    for t in range(len(val)):

        model = ARIMA(history, order = arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(val[t])

    error = rmsle(val, np.array(predictions))

    return error



def get_arima_order(dataset, p_values, d_values, q_values, target):

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    rmsle = evaluate_arima_model(dataset, order)

                    if rmsle < best_score:

                        best_score, best_cfg = rmsle, order

                except:

                    continue

    print('>>>', target, 'Best ARIMA%s RMSLE = %.3f' % (best_cfg, best_score))

    return best_cfg
train[train['Country_Region'] == 'Senegal']
p_values = [0, 1, 2, 3]

d_values = range(0, 2)

q_values = range(0, 2)

def model_per_country(country, province, show_plots = False):

    country_df = get_country(train, country, province)

    country_df_test = get_country(test, country, province)



    arima_order_cc = get_arima_order(country_df['ConfirmedCases'].values, p_values, d_values, q_values, 'ConfirmedCases')

    if country == 'Senegal':

        arima_order_cc = (1, 1, 0)

    model_cc = ARIMA(country_df.ConfirmedCases, order = arima_order_cc)

    fitted_model_cc = model_cc.fit(disp = 0)



    arima_order_fa = get_arima_order(country_df['Fatalities'].values, p_values, d_values, q_values, 'Fatalities')

    if country == 'Senegal':

        arima_order_fa = (1, 1, 0)

    model_fa = ARIMA(country_df['Fatalities'], order = arima_order_fa)

    fitted_model_fa = model_fa.fit(disp = 0)



    start_lim = len(country_df) - 10

    end_lim = start_lim + len(country_df_test) - 1



    cc_hat = fitted_model_cc.predict(start_lim, end_lim, typ= 'levels')

    fa_hat = fitted_model_fa.predict(start_lim, end_lim, typ= 'levels')



    forcast_id = country_df_test['ForecastId'].values.tolist()

    predict_res = pd.DataFrame(columns = submission.columns)

    predict_res['ForecastId'] = forcast_id

    predict_res['ConfirmedCases'] = cc_hat.values

    predict_res['Fatalities'] = fa_hat.values

   

    if show_plots:

        fitted_model_cc.plot_predict(dynamic = False, )

        fitted_model_fa.plot_predict(dynamic = False, )

        fitted_model_cc.plot_predict(arima_order_cc[1], len(country_df) + 33)

        fitted_model_fa.plot_predict(arima_order_fa[1], len(country_df) + 33)

        plt.show()

    return predict_res

    
%%time

show_plots = True

country = 'Senegal'

province = '_PROVINCE_' + country

predict_res = model_per_country(country, province, True)
%%time

submission = pd.DataFrame(columns = submission.columns)

for country in train['Country_Region'].unique():

    print('> Modeling for country:', country)

    provinces = train[train['Country_Region'] == country]['Province_State'].unique()

    for province in provinces:

        print('>> Modeling for province:', province)

        predict_res = model_per_country(country, province, False)

        submission = submission.append(predict_res)

    print()

    print()



for col in ['ConfirmedCases', 'Fatalities']:

    submission.loc[submission[col] < 0, col] = 0



submission.to_csv('submission' + '.csv', index = 0)