# Alan Vitullo
# Sunspot analysis

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #

# Input data files are available in the "../input/" directory.
# Import csv file
df = pd.read_csv("../input/Sunspots.csv")
print(df.dtypes)
print(df.head)
# Set figure width and height
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size

# Graph the number of sunspots, the source data is sorted
df.iloc[:,2:].plot(linestyle = '', marker = '.', color='b')
df.iloc[:,2:].plot(linestyle = '-', marker = '', color='b')
# Initial Scrape
# Farm out columns 
npx = np.array(df.iloc[:,:1])
npy = np.array(df.iloc[:,2:])
#Obtain base features
xf = npx.size
yf = npy.size
ymax = int(np.amax(npy))
ymean = np.mean(npy)

# Plot Figure 
# Figure Structure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.axis('tight')
# Major ticks every 60, minor ticks every 12
x_major_ticks = np.arange(0, xf, 144)
x_minor_ticks = np.arange(0, xf, 12)
y_major_ticks = np.arange(0, ymax, 25)
y_minor_ticks = np.arange(0, ymax, 5)
# And a corresponding grid
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.25)
ax.grid(which='major', alpha=0.75)
# Graph properties
ax.plot(npx, npy, color='blue')
ax.axhline(y=ymean, linewidth=3, color='black')
ax.set_ylabel('Mean Number of Sun Spots')
ax.set_xlabel('Months since -- 1749')
ax.set_title('Monthly Mean Total Sunspots');
plt.show() #Display
plt.savefig('sunspots-signal_graph.png', dpi = 200) #Output
from numpy import fft #

# Singal Data
sample_rate = 12 # one sample per month to extract yearly trends
elapsed_years = 2016 - 1749
# Sampling frequency/ num of samples
resolution = sample_rate / yf
t_step = 1 / sample_rate

# RFFT Transform to frequency space
sig = fft.rfft(npy)
freq = fft.fftfreq(yf, t_step)
N = int(yf/2+1)
ind = np.arange(1, N)
print(ind)
psd = abs(sig[ind])**2+abs(sig[-ind])**2
#c_psd = psd[ind] / sample_rate

# 
print(psd.size)
print(psd)
print(freq[ind])

#
figf = plt.figure()
afft = figf.add_subplot(1, 1, 1)

afft.plot(freq[ind], psd)
afft.plot(freq[-ind], psd)
plt.show()
plt.savefig('sunspots-frequency_graph.png', dpi = 200) #Output