# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

np.set_printoptions(suppress=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class SISModel:

    def __init__(self, dataset, total_pop=10000000):

        self.dataset = dataset

        self.params = {}

        self.total_pop = total_pop

        

    def create_simulation(self, country, region, days, offset=None):

        place_key = '_'.join([country, region])

        

        if place_key not in self.params.keys():

            beta, gamma = self.find_parameters(country, region)

            self.params[place_key] = {}

            self.params[place_key]['beta'], self.params[place_key]['gamma'] = beta, gamma

        

        infected_forecast = self.simulate(country, region, days, self.params[place_key]['beta'], self.params[place_key]['gamma'], offset)

        infected_actual = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

        

        return infected_forecast, infected_actual

            

            

    def find_parameters(self, country, region, occurrance=None):

        from scipy.optimize import minimize

        from numpy import linalg as LA

        

        self.occurrance = occurrance

        

        def rss_infection_forecast(params):

            if self.occurrance is None:

                occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

            else:

                occurrance = self.occurrance

            cases = occurrance != 0

            occurrance = occurrance[cases]

            N = self.total_pop

            I = occurrance[0]

            S = N - I

            beta, gamma = params[0], params[1]

            sus, inf = [], []

            for t in range (0, len(occurrance)):

                S = S - (beta*S*I/N) + gamma * I

                I = I + (beta*S*I/N) - gamma * I



                sus.append(S)

                inf.append(I)

            return LA.norm(np.array(inf) - occurrance, ord=2)

        

        params = np.array([0.1, 0.1])

        res = minimize(rss_infection_forecast, params, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})

        return res.x[0], res.x[1]

        

            

    def simulate(self, country, region, days, beta, gamma, offset=None):

        occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

        cases = occurrance != 0

        occurrance_zero, occurance_non_zero = occurrance[~cases], occurrance[cases]

        N = self.total_pop

        if offset:

            I = occurance_non_zero[-1]

        else:

            I = occurance_non_zero[0]

            offset = len(occurance_non_zero)

        S = N - I

        sus, inf = [], []

        for t in range (0, offset):

            S = S - (beta*S*I/N) + gamma * I

            I = I + (beta*S*I/N) - gamma * I



            sus.append(S)

            inf.append(I)

        if offset:

            return inf

        else:

            return list(occurrance_zero) + inf

    

    def simulate_future(self, country, region, days, observed=None, restimate=False, plot=False):

        if restimate:

            assert len(observed) == days, 'These should be equal'

            original_occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

            occurrance = original_occurrance

            place_key = '_'.join([country, region])

            if plot:

                boxDim = int(self.getSubplotBoxSize(days))

                fig, axs = plt.subplots(boxDim, boxDim, figsize=(20, 20))

                fig.suptitle('COVID-19 progression in {} with course correction over {} days'.format(', '.join([region, country]), str(days)), y=0.92, fontsize=20, verticalalignment='bottom')

                

                last_d = -1

            for d in range(1, days+1):

                beta, gamma = self.find_parameters(country, region, occurrance)

                self.params[place_key] = {}

                self.params[place_key]['beta'], self.params[place_key]['gamma'] = beta, gamma

                

                forecast, _ = self.create_simulation(country, region, len(occurrance), days)

                if plot:

                    row, col = int(np.floor((d-1) / boxDim)), int((d-1) % boxDim)

                    axs[row, col].plot(observed, color='b')

                    axs[row, col].plot(forecast, color='r')

                    axs[row, col].set_title('Day {}'.format(d))

                    last_d = d

                occurrance = np.hstack((original_occurrance, observed[:d]))

            if plot:

                for i in range(last_d+1, boxDim**2+1):

                    row, col = int(np.floor((i-1) / boxDim)), int((i-1) % boxDim)

                    axs[row, col].axis('off')

        else:

            forecast, _ = self.create_simulation(country, region, 77, days)

        return forecast

    

    def getSubplotBoxSize(self, days):

        return np.ceil(np.sqrt(days))
all_forecasts = {}



country = 'China'

region = 'Hong Kong'

dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

sis_model = SISModel(dataset)

test = pd.read_csv('/kaggle/input/corona-new/train.csv')

observed_all = test.loc[test['Country_Region'] == country].loc[test['Province_State'] == region]['ConfirmedCases'].to_numpy()

forecast_days = 7

observed = observed_all[-12:-(12-forecast_days)]

import matplotlib.pyplot as plt

infected_f = sis_model.simulate_future(country, region, forecast_days, observed=observed, restimate=True, plot=True)

plt.plot(infected_f, color='b')

# plt.plot(infected_o, color='r')

plt.plot(observed, color='r')

plt.show()
country = 'China'

dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

regions = np.unique(dataset.loc[dataset['Country_Region'] == country]['Province_State'].to_numpy())



dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

sis_model = SISModel(dataset)

test = pd.read_csv('/kaggle/input/corona-new/train.csv')



forecast_days = 7



for region in regions:

    observed_all = test.loc[test['Country_Region'] == country].loc[test['Province_State'] == region]['ConfirmedCases'].to_numpy()

    observed = observed_all[-12:-(12-forecast_days)]

    

    infected_f = sis_model.simulate_future(country, region, forecast_days, observed=observed, restimate=False, plot=False)

    

    # calculate errors

    mbe = np.mean(infected_f - observed)

    mae = np.mean(np.abs(infected_f - observed))

    rmse = np.sqrt(np.mean((infected_f - observed)**2))

    

    # calculate max cases recorded

    max_cases = np.max(observed)

    min_cases = np.min(observed)

    range_cases = ', '.join([str(min_cases), str(max_cases)])

    

    # print the error to the console

    print('{}, {} : MBE : {}, MAE : {}, RMSE : {}, Range of Cases : ({})'.format(region, country, mbe, mae, rmse, range_cases))
np.ceil(np.sqrt(20))
class HillModel:

    def __init__(self, dataset):

        self.dataset = dataset

        self.params = {}

        

    def create_simulation(self, country, region, days, offset=None):

        place_key = '_'.join([country, region])

        

        if place_key not in self.params.keys():

            y_ss, k, n = self.find_parameters(country, region)

            self.params[place_key] = {}

            self.params[place_key]['y_ss'], self.params[place_key]['k'], self.params[place_key]['n'] = y_ss, k, n

            print(y_ss, k, n)

        

        infected_forecast = self.simulate(country, region, days, self.params[place_key]['y_ss'], self.params[place_key]['k'], self.params[place_key]['n'], offset)

        infected_actual = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

        

        return infected_forecast, infected_actual

            

            

    def find_parameters(self, country, region, occurrance=None):

        from scipy.optimize import minimize

        from numpy import linalg as LA

        

        self.occurrance = occurrance

        

        def rss_infection_forecast(params):

            if self.occurrance is None:

                occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

            else:

                occurrance = self.occurrance

            cases = occurrance != 0

            occurrance = occurrance[cases]

            t = 1

            I = occurrance[0]

            y_ss, k, n = params[0], params[1], params[2]

            inf = [I]

            for t in range (1, len(occurrance)):

                I = I + (y_ss) / (1 + (k/t)**n)

                inf.append(I)

            return LA.norm(np.array(inf) - occurrance, ord=2)

        

        params = np.array([0.1, 0.1, 0.1]) #np.random.rand(3)

        res = minimize(rss_infection_forecast, params, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})

        return res.x[0], res.x[1], res.x[2]

        

            

    def simulate(self, country, region, days, y_ss, k, n, offset=None):

        occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

        cases = occurrance != 0

        occurrance_zero, occurance_non_zero = occurrance[~cases], occurrance[cases]

        if offset:

            t_start = len(occurance_non_zero)

        else:

            t_start = 1

            offset = len(occurance_non_zero)

            

        inf = []

        I = occurance_non_zero[t_start-1]

        for t in range (t_start, t_start+offset-1):

            I = I + (y_ss) / (1 + (k/t)**n)

            inf.append(I)

            

        if offset:

            return inf

        else:

            return list(occurrance_zero) + inf

    

    def simulate_future(self, country, region, days, observed=None, restimate=False, plot=False):

        if restimate:

            assert len(observed) == days, 'These should be equal'

            original_occurrance = self.dataset.loc[self.dataset['Country_Region'] == country].loc[self.dataset['Province_State'] == region]['ConfirmedCases'].to_numpy()

            occurrance = original_occurrance

            place_key = '_'.join([country, region])

            if plot:

                boxDim = int(self.getSubplotBoxSize(days))

                fig, axs = plt.subplots(boxDim, boxDim, figsize=(20, 20))

                fig.suptitle('COVID-19 progression in {} with course correction over {} days'.format(', '.join([region, country]), str(days)), y=0.92, fontsize=20, verticalalignment='bottom')

                

                last_d = -1

            for d in range(1, days+1):

                y_ss, k, n = self.find_parameters(country, region, occurrance)

                self.params[place_key] = {}

                self.params[place_key]['y_ss'], self.params[place_key]['k'], self.params[place_key]['n'] = y_ss, k, n

                

                forecast, _ = self.create_simulation(country, region, len(occurrance), days)

                if plot:

                    row, col = int(np.floor((d-1) / boxDim)), int((d-1) % boxDim)

                    axs[row, col].plot(observed, color='b')

                    axs[row, col].plot(forecast, color='r')

                    axs[row, col].set_title('Day {}'.format(d))

                    last_d = d

                occurrance = np.hstack((original_occurrance, observed[:d]))

            if plot:

                for i in range(last_d+1, boxDim**2+1):

                    row, col = int(np.floor((i-1) / boxDim)), int((i-1) % boxDim)

                    axs[row, col].axis('off')

        else:

            forecast, _ = self.create_simulation(country, region, 77, days)

        return forecast

    

    def getSubplotBoxSize(self, days):

        return np.ceil(np.sqrt(days))
country = 'China'

region = 'Hunan'

dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

hill_model = HillModel(dataset)

test = pd.read_csv('/kaggle/input/corona-new/train.csv')

observed_all = test.loc[test['Country_Region'] == country].loc[test['Province_State'] == region]['ConfirmedCases'].to_numpy()

forecast_days = 7

observed = observed_all[-12:-(12-forecast_days)]

import matplotlib.pyplot as plt

infected_f, infected_a = hill_model.create_simulation(country, region, 80)

plt.plot(infected_f, color='b')

plt.plot(infected_a, color='r')

# plt.plot(observed, color='r')

plt.show()
from scipy.optimize import minimize

from numpy import linalg as LA



country = 'US'

region = 'New York'

dataset = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

occurrance = test.loc[test['Country_Region'] == country].loc[test['Province_State'] == region]['ConfirmedCases'].to_numpy()

normalizer = 10000

occurrance = occurrance / normalizer



def rss_infection_forecast(params):

    t = 1

    y_ss, k, n = params[0], params[1], params[2]

    inf = []

    for t in range (1, len(occurrance)+1):

        I = (y_ss) / (1 + (k/t)**n)

        inf.append(I)

    return LA.norm(np.array(inf) - occurrance, ord=2)



params = np.array([0, 0, 0]) #np.random.rand(3)

res = minimize(rss_infection_forecast, params, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})

# res = minimize(rss_infection_forecast, params, method='BFGS', options={'disp': False})

print(res.x[0], res.x[1], res.x[2])

y_ss, k, n = res.x[0], res.x[1], res.x[2]

inf = []

for t in range (1, len(occurrance)+1):

    I = (y_ss) / (1 + (k/t)**n)

    inf.append(I * normalizer)

plt.plot(inf, color='r')

plt.plot(occurrance*normalizer, color='b')

plt.show()