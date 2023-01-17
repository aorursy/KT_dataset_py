# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

train = pd.read_csv("../input/train.csv")
                    #, dtype={"Age": np.float64}, )
    
test = pd.read_csv("../input/test.csv")
df = pd.read_csv("../input/train.csv")
train.head()
df = pd.read_csv("../input/train.csv")
df = df.drop(['Ticket','Cabin'], axis=1) 
#df = df.dropna()


fig = plt.figure(figsize = (18,6), dpi = 1600)
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.55
ax1 = plt.subplot2grid((2,2), (0,0))
df.Survived.value_counts().plot(kind = 'bar', alpha = alpha_bar_chart)
ax1.set_xlim(-1,2)
plt.title("distribution...")
plt.subplot2grid((2,2), (0,0))
