# idx [1..14] of the earthquake you'd like to animate

# first (0) and last (15) are note full cycles!!!

EARTHQUAKE = 1



# Datapoints inside the window; Set this lower if you'd like to zoom in.

WINDOW_SIZE = 150000



# Window step size

STEP_SIZE = WINDOW_SIZE // 5



# Refresh interval; lower=faster animation

REFRESH_INTERVAL = 100
import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.animation as animation



from IPython.display import HTML

from scipy import signal



earthquakes = [5656574, 50085878, 104677356, 138772453, 187641820, 218652630, 245829585, 307838917,

               338276287, 375377848, 419368880, 461811623, 495800225, 528777115, 585568144, 621985673]



train_df = pd.read_csv('../input/train.csv', nrows=earthquakes[EARTHQUAKE + 1] - earthquakes[EARTHQUAKE],

                       skiprows = earthquakes[EARTHQUAKE] + 1,

                       names=['acoustic_data', 'ttf'],

                       dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},)
%matplotlib inline



ad = train_df.acoustic_data.values#[::5]

ttf = train_df.ttf.values#[::5]



def animate(i):

    line.set_ydata(ad[i*STEP_SIZE:i*STEP_SIZE + WINDOW_SIZE])

    ax.set_xlabel('TTF {0:.8f}'.format(ttf[i*STEP_SIZE + WINDOW_SIZE]))



fig, ax = plt.subplots(figsize=(9,4))

ax.set(xlim=(0,WINDOW_SIZE), ylim=(-175, 175))



line = ax.plot(train_df.iloc[0:WINDOW_SIZE], lw=1)[0]

anim = matplotlib.animation.FuncAnimation(fig, animate, frames=(len(ad) - WINDOW_SIZE) // STEP_SIZE,

                                          interval=REFRESH_INTERVAL, repeat=True)



# You can save the animation

# anim.save('acoustic_data.gif', writer='imagemagick')



# Show the animation (does not work in kernel)

# plt.show()



# Remove this plt.close() in your experiment.

plt.close()
# Low, High "bandpass" frequencies

FREQUENCY_BAND = (45000, 55000)
%matplotlib inline



def animate_freqs(i):

    x = ad[i*STEP_SIZE:i*STEP_SIZE + WINDOW_SIZE]

    frequencies, power_spectrum = signal.periodogram(x, 4000000, scaling='spectrum')

    idx = (frequencies >= FREQUENCY_BAND[0]) & (frequencies <= FREQUENCY_BAND[1])

    line.set_data(frequencies[idx].astype(np.int32), power_spectrum[idx])

    ax.set_xlabel('TTF {0:.8f}'.format(ttf[i*STEP_SIZE + WINDOW_SIZE]))



fig, ax = plt.subplots(figsize=(9,4))



# !!!! If you don't see anything on the plot, try to adjust the `ylim` argument !!!!

ax.set(xlim=FREQUENCY_BAND, ylim=(0, .005))

ax.set_title("{}Hz - {}Hz".format(FREQUENCY_BAND[0], FREQUENCY_BAND[1]))



frequencies, power_spectrum = signal.periodogram(ad[0:WINDOW_SIZE], 4000000, scaling='spectrum')      

idx = (frequencies >= FREQUENCY_BAND[0]) & (frequencies <= FREQUENCY_BAND[1])

line = ax.plot(frequencies[idx].astype(np.int32), power_spectrum[idx], lw=1)[0]



anim = matplotlib.animation.FuncAnimation(fig, animate_freqs, frames=(len(ad) - WINDOW_SIZE) // STEP_SIZE,

                                          interval=REFRESH_INTERVAL, repeat=True)



# You can save the animation

# anim.save('frequency_{}_{}.gif'.format(FREQUENCY_BAND[0], FREQUENCY_BAND[1]), writer='imagemagick')





# Show the animation (does not work in kernel)

# plt.show()



# Remove this plt.close() in your experiment.

plt.close()
# This should help to adjust the ylim argument.

frequencies, power_spectrum = signal.periodogram(ad[0:WINDOW_SIZE], 4000000, scaling='spectrum')

idx = (frequencies >= FREQUENCY_BAND[0]) & (frequencies <= FREQUENCY_BAND[1])

pd.Series(power_spectrum[idx]).describe()