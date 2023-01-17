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
import seaborn as sn

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

curve = pd.read_csv("../input/curve.csv")

def fit_poly(degree):

    p=np.polyfit(curve.x,curve.y,deg=degree)

    curve['fit']=np.polyval(p,curve.x)

    sn.regplot(curve.x,curve.y,fit_reg=False)

    return plt.plot(curve.x,curve.fit,label='fit')

fit_poly(2)

plt.xlabel("x.value")

plt.ylabel("y.value")

import pandas as pd

curve = pd.read_csv("../input/curve.csv")