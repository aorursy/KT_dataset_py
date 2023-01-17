import pandas as pd

speed_df = pd.read_csv('../input/scafencerspeed/thrust_speed_data/fencerspeeddata.csv')

speeds = []

for i in range(len(speed_df['avgSpeed'])):

    speeds.append(speed_df['avgSpeed'][i])

len(speeds)
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import numpy

plt.rcParams['figure.figsize'] = [10,8]

fig, ax1 = plt.subplots()

ax1.hist(speeds, bins=10)

avg = numpy.mean(speeds)



ax1.axvline(avg,color='black', label="Average")

std = numpy.std(speeds)



ax1.axvline(avg + std,color='red', label="Standard deviation")

ax1.axvline(avg - std,color='red')

ax1.axvline(2000,color='green',label="Swordbot target speed")

plt.title('Histogram: Average speed of jabs with the primary hand (mm/s)')

ax1.legend(loc="upper right")

ax1.set_xlabel('speed (mm/s)')

ax1.set_ylabel('count')

fig.patch.set_facecolor('white')
