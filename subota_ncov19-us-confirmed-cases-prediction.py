!cd COVID-19 && git pull || git clone --depth 1 --single-branch https://github.com/CSSEGISandData/COVID-19.git COVID-19

!curl https://covidtracking.com/api/v1/states/daily.csv -o tests-daily.csv

    

#!curl https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/testing-in-us.html -o testing-in-us.html
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FuncFormatter

import numpy as np

import scipy, scipy.optimize, scipy.stats

import os, re

import doctest

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg')
def read_hopkins_files():

    location = './COVID-19/csse_covid_19_data/csse_covid_19_daily_reports'



    frames = []

    for root, dirs, files in os.walk(location):

        for file_name in files:

            if '.csv' in file_name:

                full_file_path = os.path.join(root, file_name)

                frame = pd.read_csv(full_file_path, index_col=None, header=0)

                frame['file_name'] = file_name

                frames.append(frame)

    for f in frames:

        f.rename(inplace=True, columns={

            'Country_Region':'Country',

            'Country/Region':'Country',

            'Province_State':'State',

            'Province/State':'State',

            'Last_Update':'LastUpdate',

            'Last Update':'LastUpdate',

            'Lat':'Latitude',

            'Long_':'Longitude'})



    frame = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    

    frame['Date'] = pd.to_datetime(

            frame.file_name.str.extract(r'(\d\d-\d\d-\d\d\d\d)', expand=False)

            , format='%m-%d-%Y')

    frame.loc[frame.file_name=='cases-web-data.csv','Date'] = frame.Date.max() + np.timedelta64(1,'D')



    frame['Active'] = frame.Confirmed - frame.Recovered - frame.Deaths



    return frame



data_hopkins = read_hopkins_files()



DATE_ZERO = data_hopkins.Date.max()

def date2day(date):

    return (date - DATE_ZERO)/np.timedelta64(1,'D')



def day2date(day):

    ''' Convert relative day number back into calendar date

    >>> day2date(date2day(pd.to_datetime('1970-06-02', format='%Y-%m-%d')))

    Timestamp('1970-06-02 00:00:00')

    >>> date2day(day2date(42))

    42.0

    '''

    return DATE_ZERO + day*np.timedelta64(1,'D')



data_hopkins['DayN'] = date2day(data_hopkins.Date)



doctest.testmod();
d = pd.read_csv('tests-daily.csv')

d.date = pd.to_datetime(d.date, format='%Y%m%d')

d.rename(columns={'date':'Date'}, inplace=True)

d['DayN'] = date2day(d.Date)

data_tracking = d

del d
class ExponentLinearModel():

    def __init__(self, x, y=None):

        if isinstance(x, pd.Series) and y==None:

            self.x, self.y = x.index, x.values

        else:

            self.x, self.y = x, y

        

        self.beta, self.pcov = scipy.optimize.curve_fit(

            self._function, self.x, self.y

            , jac = self._jacobian

            , p0 = self._default_beta())



    def best_estimate(self,x):

        return self._function(x, *self.beta)

    

    def quantile_estimate(self, x, percentiles):

        beta_sample = scipy.stats.multivariate_normal(

            mean=self.beta, cov=self.pcov, allow_singular=True).rvs(size=1000)



        y_sample = self._function(

            x, *(beta_sample.T.reshape( (len(self.beta),-1,1) ))

        )



        return np.quantile(y_sample, [0.025,0.975], axis=0)



    def conclusion(self):

        log2 = np.log(2)

        b1 = self.beta[1]

        b1sigma = np.sqrt(self.pcov[1,1])

        

        doubles_min = round(log2/(b1+b1sigma*1.95),1)

        doubles_max = round(log2/(b1-b1sigma*1.95),1)



        return f'{doubles_min:.3}..{doubles_max:.3} days to double'



    def plot(self, before=0, after=0, **kwarg):

        x = self.x

        fit_x = np.linspace(x.min()-before, x.max()+after)



        fit_y_best = self.best_estimate(fit_x)

        fit_y_min, fit_y_max = self.quantile_estimate(fit_x, [0.025,0.975])



        plt.plot(fit_x, fit_y_best, linewidth=0.5, zorder=-1, **kwarg)



        plt.fill_between(fit_x, fit_y_max, fit_y_min, alpha=.25, zorder=-10

                        , label = self.conclusion(), **kwarg)



        if before>0:

            forecast_x = np.linspace(x.min()-before, x.min())

            plt.plot(forecast_x, self.best_estimate(forecast_x)

                     ,'--', linewidth=1.5, zorder=-5, **kwarg)

        if after:

            forecast_x = np.linspace(x.max(), x.max()+after)

            plt.plot(forecast_x, self.best_estimate(forecast_x)

                     ,'--', linewidth=1.5, zorder=-5, **kwarg)



    def _function(self, x, *beta):

        return np.exp( x*beta[1] + beta[0] )



    def _jacobian(self, x, *beta):

        exp = np.exp(x*beta[1] + beta[0])

        return np.transpose([exp, exp*x])



    def _default_beta(self):

        return (0,0)



class ExponentQuadraticModel(ExponentLinearModel):

    def conclusion(self):

        if self.beta[2]>=0:

            return 'Unbound growth'

        else:

            zero_deriv = - self.beta[1]/(self.beta[2]*2)

            return f'might peak on {day2date(round(zero_deriv)):%m/%d}'



    def _function(self, x, *beta):

        return np.exp(x**2*beta[2] + x*beta[1] + beta[0])

    

    def _jacobian(self, x, *beta):

        exp = np.exp( x**2*beta[2] + x*beta[1] + beta[0] )

        return np.transpose([exp, exp*x, exp*x**2])



    def _default_beta(self):

        return (0,0,0)



def plot_fact(data, **kwarg):

    if not 'label' in kwarg:

        kwarg['label'] = data.name

    

    plt.vlines(data.index, 0 ,data.values, linestyle='dotted', color='grey')



    return plt.plot(data.index, data.values, 'o', **kwarg)



def format_axes():

    def format_people(n):

        if n< 1e3:

            div, suffix = 1, ''

        elif n< 1e6:

            div, suffix = 1e3, 'K'

        elif n < 1e9:

            div, suffix = 1e6, 'M'

        elif n < 1e12:

            div, suffix = 1e9, 'G'

        elif n < 1e15:

            div, suffix = 1e12, 'T'

        else:

            div, suffix = 1, ''

        return f'{int(n/div)}{suffix}'



    plt.xlabel('Date')



    xmin, xmax = plt.gca().get_xlim()

    step_x = max(1, int(xmax-xmin)//10)

    

    plt.gca().xaxis.set_major_locator(MultipleLocator( step_x ))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(

        lambda x, pos: day2date(x).strftime('$_{%m}%d$') ))



    plt.gca().yaxis.set_major_formatter(FuncFormatter(

        lambda y, pos: format_people(y) ))



    plt.grid(True, axis='y', which='both')

    plt.legend(loc='upper left')
d = data_hopkins.query(

    "Country == 'US' & ('2020-03-07' <= Date <= '2020-04-03')"

).groupby('DayN').Active.sum()



plt.gca().set_yscale('log')

plt.gcf().set_size_inches(12,7)

plt.title('United States, Active Cases')



america1, america2, america3 = d.iloc[:12], d.iloc[12:21], d.iloc[21:]



ExponentQuadraticModel(pd.concat([america2,america3])).plot(before=13, after=7, color='magenta')



p = plot_fact(america1, label='US 1')

ExponentLinearModel(america2.iloc[1:]).plot(before=13, after=13, color=p[0]._color)



p = plot_fact(america2, label='US 2')

ExponentLinearModel(america1.iloc[:]).plot(after=22, color=p[0]._color)



p = plot_fact(america3, label='US 3')

ExponentLinearModel(america3).plot(before=21, after=7, color=p[0]._color)



d = data_tracking[['DayN','Date','onVentilatorCurrently','hospitalizedCurrently']].query(

    "'2020-03-26' <= Date <= '2020-04-03'"

).groupby('DayN').sum()

p = plot_fact(d.hospitalizedCurrently, label='hospitalized')

ExponentLinearModel(

    d.hospitalizedCurrently

).plot(before=13, after=7, color=p[0]._color)



format_axes()

del p
p = plot_fact(d.onVentilatorCurrently, label='on vent(?)')

format_axes()
plt.gcf().set_size_inches(10,8);



for i, country_list in enumerate([

    ['US', 'Italy', 'Germany']

    , ['United Kingdom', 'Switzerland', 'Canada']

    , ['Korea, South', 'China', 'Japan']

    , ['Ukraine', 'North Macedonia']

]):

    plt.subplot(2,2,i+1)

    for country in country_list:

        d = data_hopkins.query(

            f"Country == '{country}' & (-8 <= DayN <= 0) "

        ).groupby('DayN').Active.sum()



        p = plot_fact(d, label=country)

        ExponentLinearModel(d.iloc[:-2]).plot(after=2, color=p[0]._color)

        format_axes()



#plt.savefig('countries.png', format='png');
plt.gcf().set_size_inches(8,10)

plt.gca().set_yscale('log')

y_min, y_max = float('inf'), float('-inf')

for i, country in enumerate([

        'US','Italy','Germany'

        ,'Switzerland','Canada','China','Ukraine'

        , 'New Zealand','North Macedonia']):



    d = data_hopkins.query(

        f"Country == '{country}' & (-18 <= DayN <= 0)"

    ).groupby('DayN').Active.sum()



    p = plot_fact(d, label=country)

    ExponentLinearModel(d.iloc[10:-2]).plot(after=20, color=p[0]._color)

    y_min = min(y_min, d.min())

    y_max = max(y_max, d.max())

plt.ylim( y_min*0.5, y_max*4 )

format_axes()

plt.legend(loc='lower right')

plt.grid(True, axis='x', which='both');
plt.gcf().set_size_inches(12,12)

for i, country in enumerate([

        'US','Canada','Germany'

        ,'Italy','Ukraine','Spain'

        ,'Switzerland', 'North Macedonia', 'New Zealand']):

    plt.subplot(3,3,i+1)



    d = data_hopkins.query(

        f"Country == '{country}' & (-8 <= DayN <= 0)"

    ).groupby('DayN').Active.sum()



    p = plot_fact(d, label=country)

    ExponentLinearModel(d.iloc[:-2]).plot(after=2, color=p[0]._color)

    plt.ylim( d.min()*0.9, d.max()*1.1 )

    format_axes()
plt.gcf().set_size_inches(12,4)

plt.gca().set_yscale('log')

d = data_hopkins.query(

    "Country == 'US' & (-6 <= DayN <= 0)"

).groupby('DayN').Active.sum()



p = plot_fact(d, label='US')

ExponentLinearModel(d).plot(after=25, color=p[0]._color)

plt.hlines(4.5e+6, 0, 25, color = 'red', label='Available ICUs Suffice')

format_axes()

plt.grid(True, axis='x', which='both');
plt.gcf().set_size_inches(14,4)

plt.gca().set_yscale('log')

d = data_hopkins.query(

    "Country == 'Ukraine' & (-8 <= DayN <= 0)"

).groupby('DayN').Active.sum()



p = plot_fact(d, label='Ukraine')

ExponentLinearModel(d.iloc[:-2]).plot(after=22, color=p[0]._color)

plt.hlines(3.5e+4, -9, 20, color = 'red', label='Available ICUs Suffice')

format_axes()

plt.grid(True, axis='x', which='both');
def plot_country(country, state=None, plot=(-10,0), fit=(None,None)

             , forecast=(0,0)

             , model=ExponentLinearModel):

    label = country

    d = data_hopkins.query(

        f"Country == '{country}' & ({plot[0]} <= DayN <= {plot[1]})"

    )

    if state != None:

        label = f'{country}, {state}'

        d = d.query(f"State == '{state}'")

    

    d = d.groupby('DayN').Active.sum()



    p = plot_fact(d, label=label)

    model(d.iloc[slice(*fit)]).plot(

        before=forecast[0], after=forecast[1], color=p[0]._color)



plt.gca().set_yscale('log')

plt.gcf().set_size_inches(16,5)

plt.hlines(4.5e+6, -10, 30, color = 'red', label='Available ICUs Suffice')

plot_country('US', plot=(-15,0), forecast=(0,30)

             , model=ExponentQuadraticModel)

plt.grid(True)

format_axes()
for states in [

        ['New York','Michigan','Texas','North Carolina'],

        ['Florida','California','Washington']

]:

    plt.gca().set_yscale('log')

    plt.gcf().set_size_inches(8,6)



    for state in states:

        plot_country('US', state, plot=(-14,0), fit=(None,-1), forecast=(0,28)

                 , model=ExponentQuadraticModel)

    format_axes()

    plt.grid(True, axis='x', which='both')

    plt.show()
plt.gca().set_yscale('log')

plt.gcf().set_size_inches(8,4)

plot_country('US', 'North Carolina', plot=(-14,0), forecast=(0,49)

             , model=ExponentQuadraticModel)

format_axes()

plt.ylim(100, 10000)

plt.savefig('NC.png',format='png')
plt.gca().set_yscale('log')

plt.gcf().set_size_inches(8,3)

d = data_hopkins.query(

    "Country == 'Switzerland' & -14 <= DayN"

).groupby('DayN').Active.sum()



p = plot_fact(d, label='Switzerland')

ExponentQuadraticModel(d.iloc[:-8]).plot(after=15, color=p[0]._color)

ExponentQuadraticModel(d.iloc[-8:]).plot(after=7, color='green')



format_axes()
plt.gca().set_yscale('log')

plt.gcf().set_size_inches(8,3)

plot_country('Germany', plot=(-14,0), forecast=(0,21)

             , model=ExponentQuadraticModel)

format_axes()

plt.grid(True, axis='x', which='both');
plt.gca().set_yscale('log')

#plt.hlines(3.5e+4, 0, 21, color = 'red', label='Available ICUs Suffice')

plt.gcf().set_size_inches(8,4)

plot_country('Ukraine', plot=(-14,0), forecast=(0,28)

             , model=ExponentQuadraticModel)

format_axes()

plt.grid(True, axis='x', which='major')

plt.savefig('Ukraine.png',format='png')
plt.gcf().set_size_inches(8,4)

plot_country('North Macedonia', plot=(-14,0), forecast=(0,35)

             , model=ExponentQuadraticModel)

format_axes()

plt.grid(True, axis='x', which='both')
plt.gcf().set_size_inches(16,4)

plot_country('Canada', plot=(-14,0), fit=(None,-2), forecast=(0,28)

             , model=ExponentQuadraticModel)

format_axes()

plt.grid(True, axis='x', which='both')
plt.gca().set_yscale('log')

plt.gcf().set_size_inches(8,3)

d = data_hopkins.query(

    "-14 <= DayN"

).groupby('DayN').Active.sum()



p = plot_fact(d, label='World')

ExponentQuadraticModel(d).plot(after=30, color=p[0]._color)



format_axes()

plt.grid(True, axis='x', which='both')
!rm -rf COVID-19