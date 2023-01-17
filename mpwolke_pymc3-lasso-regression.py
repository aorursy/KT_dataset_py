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
%matplotlib inline

from matplotlib.pylab import *

from pymc3 import *

import numpy as np



d = np.random.normal(size=(3, 30))

d1 = d[0] + 4

d2 = d[1] + 4

yd = .2*d1 +.3*d2 + d[2]
lam = 3



with Model() as model:

    s = Exponential('s', 1)

    tau = Uniform('tau', 0, 1000)

    b = lam * tau

    m1 = Laplace('m1', 0, b)

    m2 = Laplace('m2', 0, b)



    p = d1*m1 + d2*m2



    y = Normal('y', mu=p, sigma=s, observed=yd)
with model:

    start = find_MAP()



    step1 = Metropolis([m1, m2])



    step2 = Slice([s, tau])



    trace = sample(10000, [step1, step2], start=start)
traceplot(trace);
hexbin(trace[m1],trace[m2], gridsize = 50)

axis('off');