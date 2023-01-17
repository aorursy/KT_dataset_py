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
import pandas as pd

import numpy as np

rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = 2* x-5 + rng.randn(50)

dataframe = pd.DataFrame({'$x': x, '$y': x})
import pandas as pd

import numpy as np

rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = 2* x-5 + rng.randn(50)

dataframe = pd.DataFrame({'$x': x, '$y': y})
dataframe
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.style.use('ggplot')



plt.scatter(x, y)

plt.show()
dataframe.corr(method='pearson')
import seaborn as sns

simple_linear_reg = sns.lmplot(data=dataframe, x = '$x', y = '$y') 

plt.show()
import seaborn as sns

residual_plot = sns.residplot(data=dataframe, x = '$x', y = '$y')

plt.show()
from statistics import mean

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')



xs = np.array([1,2,3,4,5], dtype=np.float64)

ys = np.array([5,4,6,5,6], dtype=np.float64)



def best_fit_slope_and_intercept(xs,ys):

    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /

         ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)

    return m, b



def squared_error(ys_orig,ys_line):

    return sum((ys_line - ys_orig) * (ys_line - ys_orig))



def coefficient_of_determination(ys_orig,ys_line):

    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = squared_error(ys_orig, ys_line)

    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr/squared_error_y_mean)

    

m, b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]



r_squared = coefficient_of_determination(ys,regression_line)

print(r_squared)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, t_test = train_test_split(x,y, test_size = 0.30)

from sklearn.linear_model import LinearRegression 

x_traindf = pd.DataFrame(x_train)

reg.fit(x_traindf,y_train)
np.reshape(dataframe, (50, 2))