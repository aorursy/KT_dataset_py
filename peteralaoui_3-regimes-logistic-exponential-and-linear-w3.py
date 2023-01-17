# Downgrades matplotlib since plotnine is not yet compatible.

!pip install -q "matplotlib==3.2.1" "plotnine==0.5.1" adjustText
import datetime

import random



import matplotlib.pyplot as plt

from mizani.breaks import date_breaks

from mizani.breaks import timedelta_breaks

from mizani.formatters import date_format

from mizani.formatters import timedelta_format

from mizani.formatters import comma_format

from mizani.transforms import log2_trans

import numpy as np

import pandas as pd

import plotnine as gg

from plotnine.themes import elements

import statsmodels.api as sm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
SUBMIT_FOR_PRIVATE_LEADERBOARD = True



if SUBMIT_FOR_PRIVATE_LEADERBOARD:

    train_end_date = '2020-04-08'

    pred_start_date = '2020-04-09'

    pred_end_date = '2020-05-07'

else:

    train_end_date = '2020-03-25'

    pred_start_date = '2020-03-26'

    pred_end_date = '2020-04-08'
df_train = pd.read_csv(

    '/kaggle/input/covid19-global-forecasting-week-3/train.csv',

    parse_dates=['Date'])

df_train = df_train[df_train.Date <= train_end_date]

df_train.groupby('Country_Region').tail(1)
df_test = pd.read_csv(

    '/kaggle/input/covid19-global-forecasting-week-3/test.csv',

    parse_dates=['Date'])

df_test = df_test[(df_test.Date >= pred_start_date) & (df_test.Date <= pred_end_date)]

df_test.tail(5)
def preprocess(_df):

    df = _df.copy()

    df['GeoEntity'] = _df['Country_Region'] + '_' + _df['Province_State'].fillna('All')

    

    # Generates the rolling means for `confirmed`.

    if 'ConfirmedCases' in df.columns:

        group_frames = []

        grouped = df.sort_values('Date').groupby('GeoEntity')

        for group in grouped.groups:

            frame = grouped.get_group(group).copy()

            frame['NewConfirmedCases'] = frame.ConfirmedCases - frame.ConfirmedCases.shift(1)

            frame['NewConfirmedCasesSmoothed'] = frame.rolling(7).NewConfirmedCases.mean()

            frame['NewConfirmedCasesGrowth'] = frame.NewConfirmedCases.pct_change()

            frame['NewConfirmedCasesGrowthSmoothed'] = frame.rolling(7).NewConfirmedCasesGrowth.mean()

            group_frames.append(frame)

        df = pd.concat(group_frames)

    

    return df
df_train = preprocess(df_train)

df_test = preprocess(df_test)
display(df_train.sample(5))

display(df_test.sample(5))
# Plots for validation.

selected_countries = ['Spain']

display(df_train.query('Country_Region in @selected_countries').groupby('GeoEntity').tail(5))

(df_train.query('Country_Region in @selected_countries')

     .groupby('GeoEntity')

     .plot(x='ConfirmedCases', y='NewConfirmedCasesSmoothed'))
# Plot new_cases vs cumulated_cases



last_update = datetime.datetime.now().strftime('%d-%m-%Y')

bg_color = '#ffffff'

font_family = 'Liberation Sans'

y_upper_bound = 5e4



selected_countries = [

    'China',

    #'South Korea',

    #'Japan',

    'Italy',

    'France',

    #'Germany',

    #'Ireland',

    'Spain',

    'Switzerland',

    'Canada',

    #'Egypt',

    'Lebanon',

    'United Arab Emirates',

    'Sweden',

    'United Kingdom',

    'United States of America',

]



df_subset = df_train.query('Country_Region in @selected_countries')

display(df_subset.groupby('GeoEntity').tail(1))





def plot_exp_growth(df_selected):

    p = (gg.ggplot(df_selected, gg.aes(x='ConfirmedCases', y='NewConfirmedCasesSmoothed'))

        + gg.geom_line(gg.aes(color='GeoEntity'),

                       size=1, alpha=1.0,

                       show_legend=False)

        + gg.geom_point(gg.aes(fill='GeoEntity'),

                        data=df_selected.groupby('GeoEntity').tail(1),

                        size=3,

                        color='black',

                        alpha=0.8,

                        show_legend=False)

        + gg.geom_text(

              gg.aes(label='GeoEntity', color='GeoEntity'),

              data=df_selected.groupby('GeoEntity').tail(1),

              family=font_family,

              fontweight='bold',

              size=10.0,

              show_legend=False,

              ha='left',

              va='bottom',

              nudge_x=0.2)

        + gg.annotate('text', x=100, y=y_upper_bound,

                      label=f'last update on {last_update}',

                      ha='left',

                      va='top',

                      nudge_y=-0.2,

                      nudge_x=0.2,

                      family=font_family,

                      size=10,

                      color='#999999')

        + gg.scale_y_continuous(

            trans=log2_trans,

            breaks=[2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6],

            limits=[1e1, y_upper_bound],

            expand=(0, 0, 0, 0),

            labels=comma_format())

        + gg.scale_x_continuous(

            trans=log2_trans,

            expand=(0.0, 0.0, 0.1, 0.0),

            breaks=[1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6],

            limits=[100, 5e5],

            labels=comma_format())

        + gg.labs(title=f'SARS-Cov-2 - New Cases as a function of Cumulative Cases',

                  x='cumulative cases', y='new cases')

        + gg.theme_minimal()

        + gg.theme(figure_size=(12, 5),

                   title=elements.element_text(ha='left', color='#454545', family=font_family),

                   plot_title=elements.element_text(family=font_family, ha='center'),

                   plot_background=elements.element_rect(fill=bg_color, color='None'),

                   panel_grid_major=elements.element_line(color='#eadcd2', size=0.5),

                   panel_grid_minor=elements.element_line(color='#eadcd2', size=0.25)))



    print(p)



    

plot_exp_growth(df_subset)
# Automatically detects which countries dropped from the exponential growth.

# NB: we define a dropper as 5 days below the max value of `new_confirmed_smooth`

dropping_geoentities = []

grouped = df_train.sort_values('Date').groupby('GeoEntity')

for group in grouped.groups:

    frame = grouped.get_group(group)

    max_idx = frame.NewConfirmedCasesSmoothed.idxmax()

    max_val = frame.NewConfirmedCasesSmoothed[max_idx]

    downtrend_count = len(frame[(frame.index > max_idx) & (frame.NewConfirmedCasesSmoothed < max_val)])

    if downtrend_count >= 5:

        dropping_geoentities.append(group)

        print(group)
df_subset = df_train.query('GeoEntity in @dropping_geoentities')



plot_exp_growth(df_subset)
# Plot curves of cases for dropping countries



df_subset = df_train.query('GeoEntity in @dropping_geoentities')



p = (gg.ggplot(df_subset, gg.aes(x='Date', y='ConfirmedCases'))

    + gg.geom_line(gg.aes(color='GeoEntity'),

                   size=1, alpha=1.0,

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.labs(title=f'SARS-Cov-2 - Confirmed Cases over time for dropping countries',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 50),

               axis_text_y=elements.element_blank()))



print(p)
# Fit a simple logistic curve, for dropping countries



from scipy.optimize import curve_fit



geoentity = 'Spain_All'

start_date = '2020-03-01'

pred_num_days =   30





def func_logistic(x, L, k, x0): 

    return L / (1.0 + np.exp(-k*(x - x0)))





def generate_predictions(_df, num_days, col_name, col_pred_name, stats=True):

    # these are the same as the scipy defaults

    initialParameters = np.array([_df[col_name].max(), 0.1, 1.0])

    if stats:

        print(f'Initial parameters: {initialParameters}')



    # curve fit the test data, ignoring warning due to initial parameter estimates

    x = range(len(_df))

    y = _df[col_name].values

    fittedParameters, pcov = curve_fit(func_logistic, x, y, initialParameters)



    y_pred = func_logistic(x, *fittedParameters) 



    SE = np.square(y_pred - y) # squared errors

    MSE = np.mean(SE) # mean squared errors

    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE

    Rsquared = 1.0 - (np.var(y_pred - y) / np.var(y))



    if stats:

        print('Parameters:', fittedParameters)

        print('RMSE:', RMSE)

        print('R-squared:', Rsquared)

        print(f'Population infected: {int(fittedParameters[0]):,}')



    date_initial = _df.Date.iloc[-1]

    x_future_dates = pd.Series([date_initial + pd.Timedelta('%d days' % d) for d in range(1, pred_num_days+1)])

    x_dates = pd.concat([_df.Date, x_future_dates])

    x = range(len(x_dates))

    y_pred = func_logistic(x, *fittedParameters) 

    

    df_y = pd.DataFrame({

        'Date': x_dates,

        'GeoEntity': _df.GeoEntity.values[0],

        col_pred_name: y_pred,

    })

    df_y = df_y.merge(_df, on=['Date', 'GeoEntity'], how='left')

    return df_y







df = df_train.query('GeoEntity == @geoentity')[df_train.Date > start_date]

try:

    df_pred = generate_predictions(df, pred_num_days, 'ConfirmedCases', 'ConfirmedCasesPredicted')

except RuntimeError:

    df_pred = None

    print('! Failed to fit logistic curve.')

    

if df_pred is not None:

    #display(df_pred)



    df_tidy = pd.melt(df_pred,

                      id_vars=['Date'],

                      value_vars=['ConfirmedCases', 'ConfirmedCasesPredicted'],

                      value_name='y',

                      var_name='series')



    p = (gg.ggplot(df_tidy)

        + gg.geom_line(gg.aes(x='Date', y='y', color='series'), size=2, alpha=0.5)

        + gg.scale_x_datetime(breaks=date_breaks('1 weeks'))

        + gg.labs(title=f'{geoentity} COVID-19 prediction of cases', x='date', y='cases predicted')

        + gg.theme(figure_size=(10, 6)))



    print(p)
import collections



geoentity = 'Spain_All'

pred_num_days = 30

lookback_days = 10



df = df_train.query('GeoEntity == @geoentity')



df_proj_upper_bounds = pd.DataFrame({

    'Date': [],

    'TotalCasesProjected': []

})



for lookback in range(lookback_days):

    if lookback == 0:

        df_past = df.copy()

        past_dt = df_past.Date.max()

    else:

        df_past = df[:-lookback].copy()

        past_dt = df_past.iloc[-1].Date



    # Get predicitons

    try:

        df_pred = generate_predictions(df_past, pred_num_days, 'ConfirmedCases', 'ConfirmedCasesPredicted',

                                       stats=False)

    except RuntimeError:

        df_pred = None

        print('! Failed to fit logistic curve.')

    

    if df_pred is not None:

        pred_30d = df_pred.ConfirmedCasesPredicted.iloc[-1]

        df_proj_upper_bounds = df_proj_upper_bounds.append(

            pd.Series({

                'Date': past_dt, 

                'TotalCasesProjected': pred_30d}),

            ignore_index=True)





display(df_proj_upper_bounds.tail())



p = (gg.ggplot(df_proj_upper_bounds,

               gg.aes(x='Date', y='TotalCasesProjected'))

    + gg.geom_point(size=2, alpha=0.5)

    + gg.stat_smooth(geom="line", size=2, alpha=0.3, se=False)

    + gg.scale_x_datetime(breaks=date_breaks('1 weeks'))

    + gg.scale_colour_brewer(palette=7)

    + gg.labs(title=f'{geoentity} COVID-19 - total cases predicted at each point in time',

              x='date', y='total confirmed estimated')

    + gg.theme_minimal()

    + gg.theme(figure_size=(10, 6)))



print(p)
pred_frames = []

df_train_dropping = df_train.query('GeoEntity in @dropping_geoentities')

grouped = df_train_dropping.groupby('GeoEntity')

for group in grouped.groups:

    frame = grouped.get_group(group)

    pred_num_days = (pd.to_datetime(pred_end_date) - frame.Date.max()).days

    

    # Predict metrics

    df_pred = pd.DataFrame({

        'GeoEntity': frame.GeoEntity.values[0],

        'Date': list(frame.Date.values) + [

            frame.Date.max() + pd.to_timedelta('%d days' % delta) for delta in range(1, pred_num_days+1)],

    })

    fit_metrics = ['ConfirmedCases', 'Fatalities']

    for metric_name in fit_metrics:

        print(f'\n--> Fit [{metric_name}] for {group}')

        pred_metric_name = metric_name + 'Predicted'

        try:

            df_pred_metric = generate_predictions(frame, pred_num_days, metric_name, pred_metric_name,

                                                  stats=True)

            df_pred = df_pred.merge(df_pred_metric[['Date', 'GeoEntity', pred_metric_name]],

                                    on=['Date', 'GeoEntity'], how='left')

        except RuntimeError:

            print(f'!!! Failed to find parameters for {metric_name}')

            dropping_geoentities.remove(group)

            df_pred = None

            break

        if len(df_pred[pd.isna(df_pred.GeoEntity)]):

            raise ValueError('Missing some GeoEntity: ' + group)

            

    if df_pred is not None:

        pred_frames.append(df_pred)

        

df_pred_agg = pd.concat(pred_frames)
# Plot curves with predicted cases for dropping countries



sample_log_geoentities = np.random.choice(df_train_dropping.GeoEntity.unique(), 20)



p = (gg.ggplot(gg.aes(x='Date'))

    + gg.geom_line(df_train_dropping.query('GeoEntity in @sample_log_geoentities'),

                   gg.aes(y='ConfirmedCases', color='GeoEntity'),

                   size=2, alpha=1.0,

                   show_legend=False)

    + gg.geom_line(df_pred_agg.query('GeoEntity in @sample_log_geoentities'),

                   gg.aes(y='ConfirmedCasesPredicted'),

                   size=1, alpha=0.5, linetype='dashed',

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.labs(title=f'SARS-Cov-2 - Confirmed Cases over time for dropping countries',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 10),

               axis_text_y=elements.element_blank()))



print(p)
# Plot curves with predictions deaths for dropping countries



sample_log_geoentities = np.random.choice(df_train_dropping.GeoEntity.unique(), 20)



p = (gg.ggplot(gg.aes(x='Date'))

    + gg.geom_line(df_train_dropping.query('GeoEntity in @sample_log_geoentities'),

                   gg.aes(y='Fatalities', color='GeoEntity'),

                   size=2, alpha=1.0,

                   show_legend=False)

    + gg.geom_line(df_pred_agg.query('GeoEntity in @sample_log_geoentities'),

                   gg.aes(y='FatalitiesPredicted'),

                   size=1, alpha=0.5, linetype='dashed',

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.labs(title=f'SARS-Cov-2 - Fatalities over time for dropping countries',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 10),

               axis_text_y=elements.element_blank()))



print(p)
# The pool of remaining countries to predict values for.

df_train_others = df_train.query('GeoEntity not in @dropping_geoentities')
df_country = df_train_others.query('GeoEntity == "US_Kansas"')

df_country = df_country[-7:]



x = np.arange(len(df_country))

x = sm.add_constant(x)

model = sm.OLS(np.log1p(df_country.ConfirmedCases.values), x)

results = model.fit()

display(results.summary())



fig, ax = plt.subplots()

fig = sm.graphics.plot_fit(results, 1, ax=ax)
rsquared_threshold = 0.60



exp_geoentities = []

grouped = df_train_others.groupby('GeoEntity')

for group in grouped.groups:

    frame = grouped.get_group(group)

    frame = frame[-7:]



    # Fit linear model on log-values

    x = np.arange(len(frame))

    x = sm.add_constant(x)

    model = sm.OLS(np.log1p(frame.ConfirmedCases.values), x)

    results = model.fit()

    

    if results.rsquared_adj > rsquared_threshold:

        exp_geoentities.append(group)

        

print(f'Exponential regime geoentities (#{len(exp_geoentities)}): {exp_geoentities}')
df_train_exp = df_train_others.query('GeoEntity in @exp_geoentities')

display(df_train_exp.groupby('GeoEntity').tail(1))
sample_exp_geoentities = np.random.choice(df_train_exp.GeoEntity.unique(), 20)



p = (gg.ggplot(gg.aes(x='Date'))

    + gg.geom_line(df_train_exp.query('GeoEntity in @sample_exp_geoentities'),

                   gg.aes(y='ConfirmedCases', color='GeoEntity'),

                   size=2, alpha=1.0,

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.labs(title=f'SARS-Cov-2 - Confirmed Cases over time for exponential regime',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 10),

               axis_text_y=elements.element_blank()))



print(p)
# Run experiments, randomly predicting the inflexion point for each country.

# Take the mean value of all experiments.

# Constraint the inflexion point to the interval: 2-30 days





class ExponentialModelWithRandomPeak(object):

    """Models all peak values."""

    

    def __init__(self, peak_range=None):

        self.peak_range = peak_range or (1, 30)

    

    def _fit(self, y_series):

        x = np.arange(len(y_series))

        x = sm.add_constant(x)

        self._model = sm.OLS(np.log1p(y_series), x)

        self._results = self._model.fit()

        

    def _predict(self, x):

        x = sm.add_constant(x)

        return np.expm1(self._results.predict(x))

    

    def _run_experiment(self, frame, col_pred_name, peak_days, pred_num_days):

        pred_num_days_extended = max(peak_days, pred_num_days)

        

        # until peak

        x1 = np.arange(len(frame), len(frame)+peak_days, 1)

        y_pred_until_peak = self._predict(x1)



        # after peak (simple symetry here)

        x2 = np.arange(len(frame)+peak_days-2, -100, -1)

        y_sym = self._predict(x2)

        y_diff = np.diff(y_sym)[1:(pred_num_days_extended - len(x1) + 1)]

        y_pred_after_peak = np.cumsum(np.abs(y_diff)) + y_pred_until_peak[-1]



        y_pred = list(y_pred_until_peak) + list(y_pred_after_peak)

        return pd.DataFrame({

            'GeoEntity': group,

            'Date': [frame.Date.max() + pd.to_timedelta('%d days' % delta) for delta in range(1, pred_num_days+1)],

            col_pred_name: y_pred[:pred_num_days],

        })

    

    def run(self, frame, col_name, col_pred_name, pred_end_date):

        """Runs all experiments."""

        if pd.to_datetime(pred_end_date) <= frame.Date.max():

            raise ValueError('The prediction end date should be in the future.')

            

        # Fits our messy exponential model.

        self._fit(frame[col_name])

        

        pred_frames = []

        pred_num_days = (pd.to_datetime(pred_end_date) - frame.Date.max()).days

        for peak_days in range(*self.peak_range):

            df = self._run_experiment(frame, col_pred_name, peak_days, pred_num_days)

            pred_frames.append(df)

        return pd.concat(pred_frames)

        

    def run_and_predict(self, frame, col_name, col_pred_name, pred_end_date):

        df_all_pred = self.run(frame, col_name, col_pred_name, pred_end_date)

        agg_fields = {

            col_pred_name: lambda x: np.quantile(x.unique(), 0.50),

            ('%s_lower' % col_pred_name): lambda x: np.quantile(x.unique(), 0.01),

            ('%s_upper' % col_pred_name): lambda x: np.quantile(x.unique(), 0.99),

        }

        #print(df_all_pred[df_all_pred['Date'] == '2020-04-18'])

        #print(df_all_pred[df_all_pred['Date'] == '2020-04-18'][col_pred_name].agg(agg_fields))

        df_pred = df_all_pred.groupby(['GeoEntity', 'Date'])[col_pred_name].agg(agg_fields).reset_index()

        # Add the last known value to start our predictions with.

        #df_pred = df_pred.append(pd.DataFrame({

        #    'GeoEntity': frame.tail(1).GeoEntity.values[0],

        #    'Date': frame.tail(1).Date.values[0],

        #    col_pred_name: frame.tail(1)[col_name].values[0],

        #    ('%s_lower' % col_pred_name): frame.tail(1)[col_name].values[0],

        #    ('%s_upper' % col_pred_name): frame.tail(1)[col_name].values[0],

        #}, index=[0]), ignore_index=True).reset_index(drop=True)

        df_pred = df_pred.sort_values('Date', ascending=True)

        return df_pred

    



print(f'Predict until {pred_end_date}')

df_pred_agg_exp = pd.DataFrame()

grouped = df_train_exp.groupby('GeoEntity')

for group in grouped.groups:

    frame = grouped.get_group(group)

    frame = frame[-7:]

    

    # Predict cases

    model = ExponentialModelWithRandomPeak(peak_range=(2, 10))

    df_pred_cases = model.run_and_predict(frame, 'ConfirmedCases', 'ConfirmedCasesPredicted', pred_end_date)

    

    # Predict fatalities

    model = ExponentialModelWithRandomPeak(peak_range=(12, 20))

    df_pred_fatalities = model.run_and_predict(frame, 'Fatalities', 'FatalitiesPredicted', pred_end_date)

    

    df_pred = pd.merge(df_pred_cases, df_pred_fatalities, on=['GeoEntity', 'Date'])

    df_pred_agg_exp = pd.concat([df_pred_agg_exp, df_pred])



df_pred_agg_exp.groupby('GeoEntity').tail(1)
sample_exp_geoentities = np.random.choice(df_train_exp.GeoEntity.unique(), 20)



p = (gg.ggplot(gg.aes(x='Date'))

    + gg.geom_line(df_train_exp.query('GeoEntity in @sample_exp_geoentities'),

                   gg.aes(y='ConfirmedCases', color='GeoEntity'),

                   size=2, alpha=1.0,

                   show_legend=False)

    + gg.geom_ribbon(df_pred_agg_exp.query('GeoEntity in @sample_exp_geoentities'),

                     gg.aes(ymin='ConfirmedCasesPredicted_lower',

                            ymax='ConfirmedCasesPredicted_upper'),

                     alpha=0.3, fill='red',

                     show_legend=False)

    + gg.geom_line(df_pred_agg_exp.query('GeoEntity in @sample_exp_geoentities'),

                   gg.aes(y='ConfirmedCasesPredicted'),

                   size=1, alpha=0.5, linetype='dashed',

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.scale_y_log10()

    + gg.labs(title=f'SARS-Cov-2 - Confirmed Cases over time for exponential regime',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 10)))



print(p)
# The pool of remaining countries to predict values for.

df_train_others = (df_train

    .query('GeoEntity not in @dropping_geoentities')

    .query('GeoEntity not in @exp_geoentities'))
pred_frames = []

grouped = df_train_others.groupby('GeoEntity')

for group in grouped.groups:

    frame = grouped.get_group(group)



    # Check if same value for 15 consecutive days

    last_15days = frame[-15:]

    if (last_15days.NewConfirmedCases == 0).all():

        df_pred = pd.DataFrame({

            'GeoEntity': group,

            'Date': df_test.query('GeoEntity == @group').Date,

            'ConfirmedCasesPredicted': frame.tail(1).ConfirmedCases.values[0],

            'FatalitiesPredicted': frame.tail(1).Fatalities.values[0],

        })

        pred_frames.append(df_pred)

        continue

            

    # Final fallback to a linear regime, since day of 1 case.

    first_case_date = frame[frame.ConfirmedCases >= 1].Date.values[0]

    frame = frame[frame.Date >= (first_case_date - pd.to_timedelta('1 day'))]

    frame = frame[-15:]  # max of 15 days for linreg

    x = np.arange(-len(frame), 0, 1)

    x = sm.add_constant(x)

    model = sm.OLS(frame.ConfirmedCases.values, x)

    results = model.fit()

    #display(results.summary())

    

    # Predict values with linear fit

    x = np.arange(len(df_test.query('GeoEntity == @group')))

    x = sm.add_constant(x)

    y_pred = results.predict(x)

    df_pred = pd.DataFrame({

        'GeoEntity': group,

        'Date': df_test.query('GeoEntity == @group').Date,

        'ConfirmedCasesPredicted': y_pred,

        'FatalitiesPredicted': 0,

    })

    pred_frames.append(df_pred)

        

df_pred_agg_lin = pd.concat(pred_frames)
#df_train_lin = df_train_others.query('GeoEntity in @lin_geoentities')

df_train_lin = df_train_others.copy()
p = (gg.ggplot(gg.aes(x='Date'))

    + gg.geom_line(df_train_lin,

                   gg.aes(y='ConfirmedCases', color='GeoEntity'),

                   size=2, alpha=1.0,

                   show_legend=False)

    + gg.geom_line(df_pred_agg_lin,

                   gg.aes(y='ConfirmedCasesPredicted'),

                   size=1, alpha=0.5, linetype='dashed',

                   show_legend=False)

    + gg.facet_wrap('GeoEntity', scales='free_y', ncol=4)

    + gg.scale_x_datetime(breaks=date_breaks('2 weeks'), labels=date_format('%W'))

    + gg.labs(title=f'SARS-Cov-2 - Confirmed Cases over time for exponential regime',

              x='week of year', y='cumulative cases')

    + gg.theme_minimal()

    + gg.theme(figure_size=(15, 10)))



print(p)
df_submission_log = df_test.merge(

    df_pred_agg[['Date', 'GeoEntity', 'ConfirmedCasesPredicted', 'FatalitiesPredicted']],

    on=['Date', 'GeoEntity'], how='inner')

df_submission_lin = df_test.merge(

    df_pred_agg_lin[['Date', 'GeoEntity', 'ConfirmedCasesPredicted', 'FatalitiesPredicted']],

    on=['Date', 'GeoEntity'], how='inner')

df_submission_exp = df_test.merge(

    df_pred_agg_exp[['Date', 'GeoEntity', 'ConfirmedCasesPredicted', 'FatalitiesPredicted']],

    on=['Date', 'GeoEntity'], how='inner')

df_submission = pd.concat([df_submission_log, df_submission_lin, df_submission_exp])

#display(df_submission.groupby('GeoEntity').head(1))



# Format submission data

df_submission['ConfirmedCases'] = df_submission['ConfirmedCasesPredicted']

df_submission['Fatalities'] = df_submission['FatalitiesPredicted']

df_submission = df_submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]

display(df_submission.sample(3))



# Merge with public leaderboard predictions

if SUBMIT_FOR_PRIVATE_LEADERBOARD:

    df_submission_public = pd.read_csv(

        '/kaggle/input/submission-for-public-leaderboard/submission_public_leaderboard.csv')

    df_submission = pd.concat([df_submission_public, df_submission])



df_submission.to_csv('submission.csv', index=False)
len(df_submission), len(df_test)