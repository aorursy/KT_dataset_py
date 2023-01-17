import os



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.animation as animation



from IPython.display import Image
states = [

  'Alabama','Alaska','Arizona','Arkansas','California','Colorado',

  'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois',

  'Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland',

  'Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana',

  'Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York',

  'North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania',

  'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah',

  'Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
dates = pd.date_range('2020-01-01', periods=366).astype(str).tolist()



index = pd.MultiIndex.from_product([dates, states], names=['Date', 'Province_State'])



data = pd.DataFrame(index=index)



actual = pd.read_csv('/kaggle/input/covid19-models-raw-data/actuals.csv', index_col='Id')

actual = actual.loc[actual['Province_State'].isin(states)]

actual = actual[['Province_State', 'Date', 'Fatalities']].set_index(['Date', 'Province_State'], drop=True)



data = data.join(actual, on=['Date', 'Province_State'])



pred_files = sorted(os.listdir('/kaggle/input/covid19-models-raw-data/'))

ihme_preds = [f.split('.')[0] for f in pred_files if f.startswith('ihme')]

lanl_preds = [f.split('.')[0] for f in pred_files if f.startswith('2020')]



for p in ihme_preds:

    try:

        forecast = pd.read_csv(f'/kaggle/input/covid19-models-raw-data/{p}.csv')[

            ['location_name', 'date_reported', 'totdea_mean']].rename(columns={'date_reported': 'Date'})

    except:

        forecast = pd.read_csv(f'/kaggle/input/covid19-models-raw-data/{p}.csv')[

            ['location_name', 'date', 'totdea_mean']].rename(columns={'date': 'Date'})



    forecast = forecast.rename(columns={'totdea_mean': p, 'location_name': 'Province_State'})

    forecast = forecast.loc[forecast['Province_State'].isin(states)].set_index(['Date', 'Province_State'], drop=True)

    data = data.join(forecast)

    



for p in lanl_preds:

    try:

        forecast = pd.read_csv(f'/kaggle/input/covid19-models-raw-data/{p}.csv')[

            ['state', 'dates', 'q.50']].rename(columns={'dates': 'Date', 'state': 'Province_State'})

    except:

        pass



    new_col = f'lanl_{p.split("_")[0]}'

    forecast = forecast.rename(columns={'q.50': new_col})

    forecast = forecast.loc[forecast['Province_State'].isin(states)].set_index(['Date', 'Province_State'], drop=True)

    data = data.join(forecast)



lanl_preds = [f.split('.')[0] for f in data.columns if f.startswith('lanl')]

    

data = data.reset_index().set_index('Date')



periods_to_plot = 170

plot_dates = pd.date_range('2020-03-15', periods=periods_to_plot).astype(str).tolist()
states_to_plot = ['Arizona', 'California','Florida','Illinois','Massachusetts',

  'New Jersey','New York','Ohio','Pennsylvania','Texas', 'Washington','Wisconsin']
n = 5



def moving_average(x, w=n):

    return np.convolve(x, np.ones(w), 'valid') / n
for state in states_to_plot:

    

    X = data.loc[(data['Province_State']==state) & (data.index.isin(plot_dates))]



    fig = plt.figure(figsize=(12,8))

    ax = plt.axes()

    line, = ax.plot(pd.to_datetime(plot_dates), [np.nan] * len(plot_dates), c='k', label='Actual', linewidth=8)



    def init():  # only required for blitting to give a clean slate.

        line.set_ydata([np.nan] * len(plot_dates))

        return line,



    hits = []

    def animate(e):

        plt.cla()

        select = X.index.isin(plot_dates[:e])

        y = X.loc[select, 'Fatalities'].values

        y.resize(len(plot_dates))

        y[y == 0] = np.nan

        ax.plot(pd.to_datetime(plot_dates[n:]), moving_average(np.diff(y)), c='k', label='Actual', linewidth=8)

        

        # IHME

        hits = []

        for pred in ihme_preds:

            if (pred.split('_')[1] in plot_dates[:e]):

                hits.append(pred)

        alpha = 1.0

        for ee, hit in enumerate(reversed(hits)):

            y = X[hit].values

            y = moving_average(np.diff(y))

            y[y == 0] = np.nan

            ax.plot(pd.to_datetime(plot_dates[n:-n]),  y[:-n], c='g', alpha=alpha, linestyle='--')

            alpha = alpha * 0.95

            

        # LANL

        hits = []

        for pred in lanl_preds:

            if (pred.split('_')[1] in plot_dates[:e]):

                hits.append(pred)

        alpha = 1.0

        for ee, hit in enumerate(reversed(hits)):

            y = X[hit].values

            y = moving_average(np.diff(y))

            y[y == 0] = np.nan

            ax.plot(pd.to_datetime(plot_dates[n:-n]), y[:-n], c='b', alpha=alpha, linestyle='--')

            alpha = alpha * 0.85         

        

        ax.set_xlim((pd.to_datetime('2020-03-15'), pd.to_datetime('2020-08-31')))

        

        fig.autofmt_xdate()

        ylim = int(np.ceil(X.select_dtypes(include='number').diff().max().max()) / 10.0) * 10    # round up to nearest 10

        ax.set_ylim((0, ylim))

        ax.grid(False)

        plt.xticks(rotation=90)

        plt.suptitle(f'{state} Daily Fatalities', fontsize=24, y=0.99)

        plt.title(f'IHME (green) and LANL (blue) Predictions\n5-day moving averages', fontsize=15)

        

        return line,



    ani = animation.FuncAnimation(

        fig, animate, init_func=init, blit=True, save_count=periods_to_plot)



    ani.save(f"{state}.gif", writer='imagemagick', fps=4)

    os.rename(f"{state}.gif", f"{state}.gif.png")

    

    Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Arizona"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "California"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Florida"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Illinois"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Massachusetts"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "New Jersey"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "New York"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Ohio"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Pennsylvania"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Texas"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Washington"

Image(filename=f"/kaggle/working/{state}.gif.png")
state = "Wisconsin"

Image(filename=f"/kaggle/working/{state}.gif.png")