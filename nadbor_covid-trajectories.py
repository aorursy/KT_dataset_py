import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import imageio

plt.style.use('seaborn')



deaths_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv').groupby('Country/Region').sum().reset_index()

cases_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv').groupby('Country/Region').sum().reset_index()
# pivot the tables for convenience

country2deaths = dict(zip(deaths_raw['Country/Region'], deaths_raw.to_numpy()[:, 3:].astype(float)))

country2cases = dict(zip(cases_raw['Country/Region'], cases_raw.to_numpy()[:, 3:].astype(float)))

dates = pd.to_datetime(deaths_raw.columns[3:])



deaths = pd.DataFrame(dict(date=dates))

cases = pd.DataFrame(dict(date=dates))

for country in country2cases.keys():

    deaths[country] = country2deaths[country]

    cases[country] = country2cases[country]
countries = ['China', 'Italy', 'US', 'Germany', 'Iran', 'Spain', 'France', 'Korea, South', 'United Kingdom']

colors = ['red', 'black', 'green', 'brown', 'darkorchid', 'cyan', 'darkslategray', 'darkorange', 'b']



y_max = 2 * max(d for country in countries for d in deaths[country])

x_max = 2 * max(c for country in countries for c in cases[country])



def draw_trajectories(date):

    fig = plt.figure(figsize=(10, 6))

    X = np.arange(1, x_max)

    plt.plot(X, X*0.002, '--', color='grey')

    plt.text(x_max, x_max * 0.002, '0.2% mortality')

    plt.plot(X, X*0.01, '--', color='grey')

    plt.text(x_max, x_max * 0.01, '1% mortality')

    plt.plot(X, X*0.05, '--', color='grey')

    plt.text(x_max, x_max * 0.05, '5% mortality')

    plt.plot(X, X*0.25, '--', color='grey')

    plt.text(y_max * 3, y_max, '25% mortality')

    

    cases_lim = cases[cases.date <= date]

    deaths_lim = deaths[deaths.date <= date]

    for country, color in zip(countries, colors):

        plt.plot(cases_lim[country], deaths_lim[country], label=country, color=color, marker='*')





    if not isinstance(date, str):

        date = str(date.date())

    plt.title('COVID-19 country trajectories as of %s' % date)

    plt.legend()

    plt.ylim(1, y_max)

    plt.xlim(1, x_max)

    plt.yscale('log')

    plt.xscale('log')

    plt.xlabel('Confirmed Cases')

    plt.ylabel('Deaths')

    plt.close()

    return fig
# plot latest data

draw_trajectories(dates[-1])
# make animation

frames = []

for date in dates:

    fig = draw_trajectories(date)

    fig.canvas.draw()       # draw the canvas, cache the renderer

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frames.append(image)



ending_frames = 20

frames = frames + ([frames[-1]] * ending_frames)
# save animation

imageio.mimsave('./trajectories.gif', frames, fps=1.5)



# display animation

from IPython.display import Image

Image(filename="trajectories.gif")