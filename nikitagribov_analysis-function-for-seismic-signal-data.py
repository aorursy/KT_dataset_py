# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32})
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
from scipy.signal import hilbert, hann, convolve
per = 5e-6
#Calculate Hilbert transform
signal = train.acoustic_data[:int(len(train)*per)]
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)

trace1 = go.Scatter(
    y = amplitude_envelope,
    name = 'amplitude_envelope'
)
data = [trace0, trace1]
layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")
#Calculate Hann func
win = hann(50)
filtered = convolve(signal, win, mode='same') / sum(win)
trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)
trace3 = go.Scatter(
    y = filtered,
    name= 'filtered'
) 


data = [trace0, trace3]

layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")
def classic_sta_lta_py(a, nsta, nlta):
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta
#Calculate STA/LTA
sta_lta = classic_sta_lta_py(signal, 50, 1000)
trace0 = go.Scatter(
    y = signal,
    name = 'signal'
)


trace4 = go.Scatter(
    y = sta_lta,
    name= 'sta_lta'
) 


data = [trace0,trace4]

layout = go.Layout(
    title = "Part acoustic_data"
)

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Part acoustic_data")