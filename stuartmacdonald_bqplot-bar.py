# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install bqplot
!jupyter nbextension enable --py --sys-prefix bqplot
import numpy as np

import bqplot.pyplot as plt

from bqplot import LinearScale, Axis, Lines, Bars, Figure
axes_options = {'x': {'label': 'x'}, 'y': {'label': 'y'}}
x = np.arange(100)

y = np.cumsum(np.random.randn(2, 100), axis=1) #two random walks



fig = plt.figure(animation_duration=1000)

lines = plt.plot(x=x, y=y, colors=['red', 'green'], axes_options=axes_options)

fig
# update data of the line mark

lines.y = np.cumsum(np.random.randn(2, 100), axis=1)
fig = plt.figure(animation_duration=1000)

x, y = np.random.rand(2, 20)

scatt = plt.scatter(x, y, colors=['blue'], axes_options=axes_options)

fig
#data updates

scatt.x = np.random.rand(20) * 10

scatt.y = np.random.rand(20)
data = np.random.rand(6)



fig = plt.figure(animation_duration=1000)

pie = plt.pie(data, radius=180, sort=False, display_labels='outside', display_values=True,

          values_format='.0%', labels=list('ABCDEFGHIJ'))

fig
pie.sizes = np.random.rand(8)
pie.sort = True
#make pie a donut

with pie.hold_sync():

    pie.radius = 180

    pie.inner_radius = 120
n = 10

x = list('ABCDEFGHIJ')

y1, y2 = np.random.rand(2, n)
fig = plt.figure(animation_duration=5000)

bar = plt.bar(x, [y1, y2], padding=0.2, type='grouped')

fig
y1, y2 = np.random.rand(2, n)

bar.y = [y1, y2]
xs = LinearScale()

ys1 = LinearScale()

ys2 = LinearScale()



x = np.arange(20)

y = np.cumsum(np.random.randn(20))

y1 = np.random.rand(20)



line = Lines(x=x, y=y, scales={'x': xs, 'y': ys1}, colors=['magenta'], marker='square')

bar = Bars(x=x, y=y1, scales={'x': xs, 'y': ys2}, colorpadding=0.2, colors=['steelblue'])



xax = Axis(scale=xs, label='x', grid_lines='solid')

yax1 = Axis(scale=ys1, orientation='vertical', tick_format='0.1f', label='y', grid_lines='solid')

yax2 = Axis(scale=ys2, orientation='vertical', side='right', tick_format='0.0%', label='y1', grid_lines='none')



Figure(marks=[bar, line], axes=[xax, yax1, yax2], animation_duration=5000)
# update mark data

line.y = np.cumsum(np.random.randn(20))

bar.y = np.random.rand(20)