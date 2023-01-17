# importing pckgs

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import csv

%matplotlib inline



df = pd.read_csv("/kaggle/input/f500-diversity/2017-F500-diversity-data.csv")
df.head(12)
df.tail()
df.shape
df['data-avail'].unique()
df.describe()
df.sample(5)
from random import sample



randomIndex = np.array(sample(range(len(df)), 5))



dfSample = df.ix[randomIndex]



print(dfSample)
print(randomIndex)
dfChoice = df.ix[0]



print(dfChoice)
pd.isnull(df)
list(df)
#Totals for white, black, hispanic, and asian females

test = df.groupby(['WHF10', 'BLKF10', 'HISPF10', 'ASIANF10'])



test.size()
df1 = df[['f500-2017-rank','name','FT10', 'FT11']] #FT10=FEMALE-TOTAL, FT11=PREVIOUS-YEAR-FEMALE-TOTAL 



df1 = df1[df1.FT10 != 'n/a']



print(df1)
df1 = df[['FT10']]



df1 = df1[df1.FT10 != 'n/a']



print(df1)
df1['FT10'].median()
plt.plot(df1)



plt.legend(['2017 Female Total'])



plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from scipy import stats 



from sklearn.datasets.samples_generator import make_blobs
X = df['f500-2017-rank'] 

y = df1
X, y = make_blobs(n_samples=100, centers= np.array([[-1],[1]]), n_features=1, shuffle=True, random_state=2017)


plt.scatter(X,y)


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X,y)

xgrid = np.linspace(-3.5,3.5, 1000)

Xgrid = xgrid.reshape(-1,1)

Y = y.reshape(-1,1)

yp = lr.predict(X)

ypgrid = lr.predict(Xgrid)


x = X.flatten()


plt.plot(X,y,'o', alpha=0.4, ms=5)

plt.plot(xgrid, ypgrid)

plt.plot(x, yp, 's', alpha=.5, ms=5)


def makeyaprob(y):

    if y >=1.0:

        return 1.0

    elif y <= 0.0:

        return 0.0

    else:

        return y

vector_makeyaprob = np.vectorize(makeyaprob)

predict_proba = lambda lr, X: vector_makeyaprob(lr.predict(X))
ypfilt = vector_makeyaprob(ypgrid)
plt.plot(X,y,'o')

plt.plot(xgrid, ypfilt)
ypred = 1*(predict_proba(lr, X) >= 0.5)
from sklearn.metrics import accuracy_score
accuracy_score(y, ypred)