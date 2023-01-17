%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set(style="darkgrid")



times = np.array([ 93.,  96.,  99., 102., 105., 108., 111., 114., 117.,

                  120., 123., 126., 129., 132., 135., 138., 141., 144.,

                  147., 150., 153., 156., 159., 162.])

temps = np.array([310.7, 308.0, 296.4, 289.5, 288.5, 287.1, 301.1, 308.3,

                  311.5, 305.1, 295.6, 292.4, 290.4, 289.1, 299.4, 307.9,

                  316.6, 293.9, 291.2, 289.8, 287.1, 285.8, 303.3, 310.])


# Create a figure

fig = plt.figure(figsize=(10, 6))



# Ask, out of a 1x1 grid, the first axes.

ax = fig.add_subplot(1, 1, 1)



# Plot times as x-variable and temperatures as y-variable

ax.plot(times, temps)
# Add some labels to the plot

ax.set_xlabel('Time')

ax.set_ylabel('Temperature')



# Prompt the notebook to re-display the figure after we modify it

fig
ax.set_title('GFS Temperature Forecast', fontdict={'size':16})



fig
# Set up more temperature data

temps_1000 = np.array([316.0, 316.3, 308.9, 304.0, 302.0, 300.8, 306.2, 309.8,

                       313.5, 313.3, 308.3, 304.9, 301.0, 299.2, 302.6, 309.0,

                       311.8, 304.7, 304.6, 301.8, 300.6, 299.9, 306.3, 311.3])
fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1, 1, 1)



# Plot two series of data

# The label argument is used when generating a legend.

ax.plot(times, temps, label='Temperature (surface)')

ax.plot(times, temps_1000, label='Temperature (1000 mb)')



# Add labels and title

ax.set_xlabel('Time')

ax.set_ylabel('Temperature')

ax.set_title('Temperature Forecast')



# Add gridlines

ax.grid(True)



# Add a legend to the upper left corner of the plot

ax.legend(loc='upper right')
fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(1, 1, 1)



# Specify how our lines should look

ax.plot(times, temps, color='tab:red', label='Temperature (surface)')

ax.plot(times, temps_1000, color='tab:green', linestyle='--',

        label='Temperature (isobaric level)')



# Same as above

ax.set_xlabel('Time')

ax.set_ylabel('Temperature')

ax.set_title('Temperature Forecast')

ax.grid(True)

ax.legend(loc='upper left')
t = np.linspace(-50,50,500)

func = [np.sin(t),np.cos(t),np.tan(t),np.exp(t),np.sin(t)+np.cos(t),np.sin(t)+np.exp(t)]

axes =[(i,j) for i in range(2) for j in range(3)]



fig,ax = plt.subplots(2, 3)

for axe,y in zip(axes,func):

    ax[axe].plot(y)

    

  
freq_x = 50

y = np.sin(2*np.pi*freq_x*t)

fft_y = np.fft.fft(y)

freq = np.fft.fftfreq(t.shape[-1])



fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].plot(y)

ax[1].plot(freq,fft_y)
import matplotlib.animation as animation



fig, ax = plt.subplots()



x = np.arange(0, 2*np.pi, 0.01)

line, = ax.plot(x, np.sin(x))





def animate(i):

    line.set_ydata(np.sin(x + i/10.0))  # update the data

    return line,





# Init only required for blitting to give a clean slate.

def init():

    line.set_ydata(np.ma.array(x, mask=True))

    return line,



ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,

                              interval=25, blit=True)



HTML(ani.to_html5_video())