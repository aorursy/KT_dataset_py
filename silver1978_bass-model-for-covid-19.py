# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class BassModel():



          

    def _add_cumulative(self):

        self.df['cum_' + self.daily] = self.df[self.daily].cumsum()

        

    def _df_for_liner_reg(self):

        self.df['N'] = self.df['cum_' + self.daily].shift(1)

        self.df['N**2'] = self.df['cum_' + self.daily].shift(1)**2

        return self.df.dropna()

        

    def fit(self, df):

        """ models takes datase as indput with date index and a daily obesrvation """

        """ first we build the cumulative of the daily observation"""

        self.df = pd.DataFrame(df)

        self.daily=self.df.columns[0]

        self._add_cumulative()

        """ second  add columns fo cumulative T-1 and cumulative T-1 squared"""

        df_for_lr = self._df_for_liner_reg()       

        lm = linear_model.LinearRegression()

        lm.fit(df_for_lr[['N', 'N**2']], df_for_lr[self.daily])

   



        """ then caluclate b0 b1 b2 and m"""

        self.b0 = lm.intercept_

        self.b1 =  lm.coef_[0]

        self.b2 =  lm.coef_[1]

        self.m = (-self.b1- np.sqrt(self.b1**2-(4*self.b0*self.b2))) /( 2 * self.b2)

        

        print ('b0: {}, b1: {}, b2: {}, m: {}'.format(self.b0, self.b1, self.b2, self.m)) 

        

    def predict(self, t):

        

        """ calculate p and q the parameter for innovator  and imitator"""

        self.p2 = self.b0/self.m

        self.q2 = -self.b2*self.m



        print ("p:{}, q: {}".format(self.p2, self.q2))

        

        predictionQt2 = [0,]

        predictionN2 = [0,]



        """ then for n dates calculate the predictions"""

        for i in range(0,t):

            N = predictionN2[i]

#             adoptorenT = (self.p2*self.m)+(self.q2-self.p2)*N + (-self.q2/self.m) * N**2

            adoptorenT = (self.p2*self.m)+(self.q2-self.p2)*N+(-self.q2/self.m)*(N**2)

            adoptorenT = round(adoptorenT, 2)

            predictionQt2.append(adoptorenT)

            predictionN2.append((predictionN2[i]+adoptorenT))

            

        return predictionQt2, predictionN2



def daily(df):

    return df- df.shift()

#  utility class for running bass model with different countries and region



class Build_Bass ():

    

    """ take the owrld dataset and """

    def __init__(self, df, country, province=None):

        self.country = country

        self.df =  df[df['Country/Region'] == country]

        if province:

            self.df = self.df[self.df['Province/State'] == province]

        self.df.set_index('Date',inplace = True)

        self.df['daily'] = daily(self.df['Confirmed'])

    

    """ run fit and predict using bass model"""

    def bassdf(self, start_date='2020-01-22', period=61):

        bm = BassModel()

        bm.fit(self.df['daily'])

        bassDF = pd.DataFrame(index=pd.date_range(start_date, periods=period, freq='d'))

        pred_pos, pred_cum = bm.predict(period-1)

        bassDF["Prediction_Positive"]= pred_pos

        bassDF["Prediction_Cumulatve"]  = pred_cum

        self.df = bassDF.join(self.df, how='outer')

        

    """ plot predticted and obeserved values"""    

    def plot(self):

        self.df['Prediction_Cumulatve'].plot(figsize=(12,8), title = self.country.upper(), legend=True)

        self.df['Confirmed'].plot(legend=True)

        self.df['Prediction_Positive'].plot(legend=True)

        self.df['daily'].plot(legend=True)

        

        

    def plot_daily(self):

        self.df['Prediction_Positive'].plot(figsize=(12,8))

        self.df['daily'].plot(figsize=(12,8))

        

import pandas as pd

nCoV_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
world = covid_19_data

world['Date'] = world['ObservationDate']
bb = Build_Bass(world, 'Mainland China', 'Hubei')

bb.bassdf(start_date='2020-01-23', period=59)

bb.plot()
bb = Build_Bass(world, 'South Korea')

bb.bassdf(start_date='2020-02-12', period=37)

bb.plot()
bb = Build_Bass(world, 'Italy')

bb.bassdf(start_date='2020-02-16', period=60)

bb.plot()
bb.plot_daily()
it = world[world['Country/Region'] == 'Italy']

it['daily'] = daily(it['Confirmed'])
bb = Build_Bass(world, 'Iran')

bb.bassdf(start_date='2020-02-22', period=31)

bb.plot()
bb = Build_Bass(world, 'Germany')

bb.bassdf(start_date='2020-02-23', period=31)

bb.plot()