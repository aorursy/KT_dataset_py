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
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
xs=np.array([1,2,3,4,5,6],dtype=np.float64)
ys=np.array([5,4,6,5,6,7],dtype=np.float64)
def best_fit_slope_and_intercept(xs,ys):
    m=(((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2)-mean(xs**2)))
    b=mean(ys)-m*mean(xs)
    return m,b
def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)**2)
def coeffecient_of_determination(ys_orig,ys_line):
    y_mean_line=mean(ys_orig)
    squared_error_regr=squared_error(ys_orig,ys_line)
    squared_error_y_mean=squared_error(ys_orig,y_mean_line)
    return 1-(squared_error_regr/squared_error_y_mean)
m,b=best_fit_slope_and_intercept(xs,ys)
#print(m,b)
regression_line=[(m*x)+b for x in xs]
predict_x=8
predict_y=(m*predict_x)+b
r_squared=coeffecient_of_determination(ys,regression_line)
print(r_squared)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y)
plt.plot(xs,regression_line)
plt.show()