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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
Time=pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv').copy()

Time.head()
Time.deceased.plot()
olum=Time.deceased.copy()

olum.index=Time.date

olum=pd.DataFrame(olum)
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(olum)

plot_pacf(olum)
#If we want to see for bigger lags autocorrelation graphic, we can use this function

from pandas.plotting import autocorrelation_plot

plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})

autocorrelation_plot(olum)
from statsmodels.tsa.ar_model import AR

model2=AR(olum).fit()#First model

model2.aic#Our aic value
model2.params#Our params
np.mean(model2.resid)
model2.sigma2
sns.distplot(model2.resid)
from scipy.stats import shapiro

shapiro(model2.resid)
forecast=Time.deceased.copy()

for i in range(1,10):

    forecast[i+70]=(model2.params[0]+forecast[len(forecast)-1]*model2.params[1]+forecast[len(forecast)-2]*model2.params[2]+forecast[len(forecast)-3]*model2.params[3]+forecast[len(forecast)-4]*model2.params[4]+forecast[len(forecast)-5]*model2.params[5]+

                    forecast[len(forecast)-6]*model2.params[6]+forecast[len(forecast)-7]*model2.params[7]+forecast[len(forecast)-8]*model2.params[8]+

                    forecast[len(forecast)-9]*model2.params[9]+forecast[len(forecast)-10]*model2.params[10]+forecast[len(forecast)-11]*model2.params[11])
forecast.tail(10)
from scipy.stats import boxcox

from scipy.special import inv_boxcox

donusum5=olum.copy()

donusum5=donusum5+1

donusum5,fitted_lambda2= boxcox(donusum5.iloc[:,0],lmbda=None)

inv_boxcox(donusum5,fitted_lambda2)
model_10=AR(donusum5).fit()

model_10.aic
model_10.params#Changed the coefficient of parameters
model_10.pvalues#When we look p values, our parameters are looking statistical significant
forecast4=donusum5.copy()

forecast4=np.array(forecast4)

for i in range(1,40):

    

    forecast4=np.append(forecast4,(model_10.params[0]+forecast4[len(forecast4)-1]*model_10.params[1]+forecast4[len(forecast4)-2]*model_10.params[2]+

                     forecast4[len(forecast4)-3]*model_10.params[3]+forecast4[len(forecast4)-4]*model_10.params[4]+forecast4[len(forecast4)-5]*model_10.params[5]+

                    forecast4[len(forecast4)-6]*model_10.params[6]+forecast4[len(forecast4)-7]*model_10.params[7]+forecast4[len(forecast4)-8]*model_10.params[8]+

                    forecast4[len(forecast4)-9]*model_10.params[9]+forecast4[len(forecast4)-10]*model_10.params[10]+forecast4[len(forecast4)-11]*model_10.params[11]))
sonuc=pd.DataFrame(inv_boxcox(forecast4,fitted_lambda2))#We converted boxcox data to original data

sonuc.tail(40)
pd.DataFrame(inv_boxcox(forecast4,fitted_lambda2)).plot()