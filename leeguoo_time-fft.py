# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def getfreq(df, tmax, nbins):

        t = list(df.time)

        t.append(tmax)

        hist, edges = np.histogram(t,bins=nbins)

        n = len(hist)

        r = np.correlate(hist, hist,mode="full")[-n:]

        sp = np.fft.fft(r)
        freq = np.fft.fftfreq(len(r))

        return freq, sp.real


df = pd.read_csv("../input/train.csv")

tmax = max(df.time)


nbins = int(tmax/60)+1
tmax = nbins*60.0


pids =  list(set(df.place_id))

npids = len(pids)

tsp = [0.0 for i in range(nbins)]

for pid in pids[:100]:

        data = df[df.place_id==pid]
        f, sp = getfreq(data, tmax, nbins)

        tsp = tsp+sp/npids

hr = [int(1.0/ff) for ff in f[1:nbins/2]]
tsp = tsp[1:nbins/2]

#############################################

t = hr

sp = tsp

ct = {}

for i, tt in enumerate(t):
        if tt in ct.keys():
                if sp[i]>ct[tt]:
                        ct[tt] = sp[i]
        else:
                ct[tt] = sp[i]
x = []
y = []
for k in sorted(ct.keys()):
        x.append(k)
        y.append(ct[k])
plt.xlim([0,1000])
plt.plot(x, y)

peaks = []
for i in range(len(y)):
        if i>0 and i<len(y)-1:
                if y[i]-y[i-1]>0 and y[i]-y[i+1]>0:
                        if y[i]-y[i-1]>30 or y[i]-y[i+1]>30:
                                peaks.append(x[i])


tmp = []

for i, peak in enumerate(reversed(peaks)):

        plt.plot([peak,peak],[5,10000],c="k",ls="--")

plt.show()