from datetime import timedelta

from numpy import linspace

from pathlib import Path

from pandas import DataFrame, Series, read_csv



FIGSIZE = (9, 5)

FOLDER = Path("../input/covid19-local-us-ca-forecasting-week-1")
train = ( 

    read_csv(FOLDER / "ca_train.csv", parse_dates=['Date'])

    ['Id Date ConfirmedCases Fatalities'.split()]

    .set_index('Date')

)

train.tail(20)
forecast_id = read_csv(FOLDER / "ca_test.csv", parse_dates=['Date']).set_index('Date')['ForecastId']

forecast_id.tail()
def contagion_window(delay=4, duration=14, reproduction=2.5):

    window = Series(linspace(1, 0, duration), name='window')

    window *= reproduction / window.sum()

    window[delay:] = 0

    

    return window.rename_axis('days since infection')



axes = contagion_window().plot.bar(color='red', figsize=FIGSIZE, title='expected transmissions')
def convolved(values, window):

    """ int or float: Discrete convolution of two timeseries. """

    return sum( w * x for w, x in zip(window, reversed(values)) )



def predict(train, dates, confirm_rate=0.2, population=40_000_000, vax_rate=0, **kwargs):

    """ DataFrame: Predicted future cases and fatalities. """



    aday = timedelta(days=1)

    data = DataFrame(index=dates)

    window = contagion_window(**kwargs)

    duration = len(window)

    new_days = data.index.difference(train.index)



    # Backfill data from training-testing overlap

    data['confirmed'] = train['ConfirmedCases']

    data['deceased'] = train['Fatalities']

    

    # Estimate infection counts and real mortality rate

    data['confirmed'] = train['ConfirmedCases']

    data['exposed'] = (train['ConfirmedCases'] / confirm_rate)

    mortality = data['deceased'].sum() / data['exposed'].sum()

    print("Estimated mortality:", round(100 * mortality, 2), "%")

 

    # Predict new cases

    data['new'] = data['exposed'].diff()

    data.at[dates.min(), 'new'] = data['exposed'][0]

    for t in new_days:

        deltas = data.loc[:t-aday, 'new']

        data.at[t, 'new'] = (1 - vax_rate - sum(deltas) / population ) * convolved(deltas, window)

    

    # Calculate cumulative totals

    data.loc[new_days, 'exposed'] = data['new'].cumsum()

    data.loc[new_days, 'deceased'] = data.loc[new_days, 'exposed'] * mortality

    data.loc[new_days, 'confirmed'] = data.loc[new_days, 'exposed'] * confirm_rate



    return data.sort_index(axis=1).astype(int)

    

data = predict(train, forecast_id.index)

axes = data.plot.line(color='bkgr', figsize=FIGSIZE, logy=True)

data.tail()
submission = DataFrame(index=data.index)

submission['ForecastId'] = forecast_id

submission['ConfirmedCases'] = data['confirmed']

submission['Fatalities'] = data['deceased']

submission.to_csv('submission.csv', index=False)