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

from bqplot import *
size = 100

np.random.seed(0)



x_data = range(size)

y_data = np.random.randn(size)

y_data_2 = np.random.randn(size)

y_data_3 = np.cumsum(np.random.randn(size) * 100.)


x_ord = OrdinalScale()

y_sc = LinearScale()



bar = Bars(x=np.arange(10), y=np.random.rand(10), scales={'x': x_ord, 'y': y_sc})

ax_x = Axis(scale=x_ord)

ax_y = Axis(scale=y_sc, tick_format='0.2f', orientation='vertical')



Figure(marks=[bar], axes=[ax_x, ax_y], padding_x=0.025, padding_y=0.025)