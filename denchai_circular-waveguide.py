# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import scipy.special as sp

import matplotlib.pyplot as plt

# plt.matplotlib.rc('text', usetex = True)

plt.matplotlib.rc('grid', linestyle = 'dotted')

plt.matplotlib.rc('figure', figsize = (6.4,4.8)) # (width,height) inches

x = np.linspace(0, 15, 500)

for v in range(0, 6):

    plt.plot(x, sp.jv(v, x))

plt.xlim((0, 15))

plt.ylim((-0.5, 1.1))

plt.legend(('${J}_0(x)$', '${J}_1(x)$', '${J}_2(x)$','${J}_3(x)$', '${J}_4(x)$', '${J}_5(x)$'), loc = 0)

plt.xlabel('$x$')

plt.ylabel('${J}_n(x)$')

plt.grid(True)

# plt.tight_layout(0.5)
import numpy as np

import scipy.special as sp

import matplotlib.pyplot as plt

# plt.matplotlib.rc('text', usetex = True)

plt.matplotlib.rc('grid', linestyle = 'dotted')

plt.matplotlib.rc('figure', figsize = (6.4,4.8)) # (width,height) inches

x = np.linspace(0, 15, 500)

for v in range(0, 6):

    plt.plot(x, 0.5*(sp.jv(v-1, x)-sp.jv(v+1, x)))

#     plt.plot(x, -sp.jv(1, x))

plt.xlim((0, 15))

plt.ylim((-0.8, 0.8))

plt.legend(('${J^\prime}_0(x)$', '${J ^\prime}_1(x)$', '${J ^\prime}_2(x)$','${J ^\prime}_3(x)$', '${J ^\prime}_4(x)$', '${J ^\prime}_5(x)$'), loc = 1)

plt.xlabel('$x$')

plt.ylabel('${J ^\prime}_n(x)$')

plt.grid(True)

# plt.tight_layout(0.5)