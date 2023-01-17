import numpy as np 

import pandas as pd 

from scipy import stats

import pylab 

import matplotlib.pyplot as plt



df_heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df_heart.columns

trestbps = df_heart['trestbps']

chol = df_heart['chol'] 
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))



skewness = stats.skew(chol)

title = f'original, skewness = {round(skewness, 2)}'

chol.plot(kind='hist', ax=ax1, color='red', alpha=0.5, title=title)





chol_t = chol.apply(np.log)

chol_t = pd.Series(chol_t)

skewness_t = stats.skew(chol_t)

title_t = f'transformed, skewness = {round(skewness_t, 2)}'

chol_t.plot(kind='hist', ax=ax2, color='cyan', alpha=0.8, title=title_t)

plt.tight_layout()

plt.show()
from statsmodels.graphics.gofplots import qqplot

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(5, 20))



# original

skewness = stats.skew(trestbps)

title = f'original, skewness = {round(skewness, 2)}'

trestbps.plot(kind='hist', ax=ax1, color='red', alpha=0.5, title=title)



##qqplot

qqplot(data=trestbps, dist="norm", ax=ax2, line='s')



# transformation

trestbps_t, lmbda_best = stats.boxcox(trestbps)

trestbps_t = pd.Series(trestbps_t)

skewness_t = stats.skew(trestbps_t)

title_t = f'transformed, skewness = {round(skewness_t, 2)}, lambda = {round(lmbda_best, 2)}'

trestbps_t.plot(kind='hist', ax=ax3, color='cyan', alpha=0.8, title=title_t)



##qqplot

qqplot(data=trestbps_t, dist="norm", ax=ax4, line='s')





plt.tight_layout()

plt.savefig('.png')

plt.show()
from sklearn.preprocessing import MinMaxScaler

from scipy.special import logit

mms = MinMaxScaler()

trestbps_mms = pd.Series(mms.fit_transform(trestbps.values.reshape(-1, 1)).flatten())





fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

title = f'original'

trestbps_mms.plot(kind='hist', ax=ax1, color='red', alpha=0.5, title=title)



trestbps_t = pd.Series(logit(trestbps_mms))

trestbps_t = trestbps_t.replace(np.Inf, 4) # for the plot

trestbps_t = trestbps_t.replace(np.NINF, -4) # for the plot

trestbps_t = pd.Series(trestbps_t)

title_t = f'transformed'

trestbps_t.plot(kind='hist', ax=ax2, color='cyan', alpha=0.8, title=title_t)

plt.tight_layout()

plt.show()