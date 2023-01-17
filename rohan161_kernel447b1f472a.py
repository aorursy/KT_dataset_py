import pandas as pd

from itertools import cycle, islice

import matplotlib.pyplot as plt

from pandas.plotting import parallel_coordinates



%matplotlib inline
P = pd.read_csv('../input/P')

P.columns = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 

        'max_wind_speed','relative_humidity','prediction']

P.head()


def parallel_plot(data):

	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))

	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])

	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
parallel_plot(P[P['relative_humidity'] < -0.5])
parallel_plot(P[P['air_temp'] > 0.5])
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])