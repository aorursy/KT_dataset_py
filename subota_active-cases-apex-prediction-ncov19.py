import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import scipy, scipy.optimize, scipy.stats
import os, re
import doctest
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
p_incub, _ = scipy.optimize.curve_fit(
            lambda x, s, a, b: scipy.stats.lognorm.cdf(x, s, a,b),
            [2.2,3.8,5.1,6.7,11.5], [0.025, 0.25, 0.5,0.75, 0.975])
incubation = scipy.stats.lognorm(*p_incub).cdf

x = np.linspace(0, 20, 100)
plt.plot(x, incubation(x), label='$$')
plt.title('COVID-19 Symptoms Onset Cumulative Distribution$^{[12]}$')
plt.xlabel('Days')
plt.ylabel('Symptoms Developed')
plt.grid(True);
plt.savefig('Incubation.png',format='png')
!cd COVID-19 && git pull || git clone --depth 1 --single-branch https://github.com/CSSEGISandData/COVID-19.git COVID-19
!curl https://covidtracking.com/api/v1/states/daily.csv -o tests-daily.csv
    
#!curl https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/testing-in-us.html -o testing-in-us.html
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
    if isinstance(date,str):
        date = pd.to_datetime(date, format='%Y-%m-%d')
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
        y_sample = self._function(
            x, *(self.beta_sample().reshape( (len(self.beta),-1,1) ))
        )

        return np.quantile(y_sample, [0.025,0.975], axis=0)
    
    def beta_sample(self, size=1000):
        return scipy.stats.multivariate_normal(
            mean=self.beta, cov=self.pcov, allow_singular=True).rvs(size=size).T

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
    '''
    f = exp(b_2x^2+b_1x+b_0)
    df/dx = exp(b_2x^2+b_1x+b_0)*(2b_2x+b_1)
    df/dx = 0  =>  x = - b_1/(2b_2)
    '''
    def conclusion(self):
        b = self.beta_sample()
        peaks = - b[1]/(b[2]*2)
        # b[2]>0 means parabola has minimum and no maximum (apex)
        peaks[b[2]>0] = float('inf')
        q_peaks = np.quantile(peaks, [0.025,0.975], axis=0)
        
        def format_peak(p):
            if p == float('inf'):
                return '$\infty$'
            else:
                return f'{day2date(p):%m/%d}'
        
        return f'peaks {format_peak(q_peaks[0])}..{format_peak(q_peaks[1])}'

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

def format_axes(yscale='log'):
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

    plt.gca().set_yscale(yscale)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(
        lambda y, pos: format_people(y) ))

    plt.grid(True, axis='y', which='both')
    plt.legend(loc='upper left')
    
def plot_country(country, state=None, plot=(-10,0), fit=(None,None)
             , forecast=(0,0)
             , model=ExponentLinearModel):
    label = country
    if isinstance(country,str):
        country = [country]
    d = data_hopkins[
        data_hopkins.Country.isin(country)
    ].query(
        f"({plot[0]} <= DayN <= {plot[1]})"
    )
    if state != None:
        label = f'{country}, {state}'
        d = d.query(f"State == '{state}'")
    
    d = d.groupby('DayN').Active.sum()

    p = plot_fact(d, label=label)
    model(d.iloc[slice(*fit)]).plot(
        before=forecast[0], after=forecast[1], color=p[0]._color)
d = data_hopkins.query(
    "Country == 'US' & ('2020-03-07' <= Date)"
).groupby('DayN').Active.sum()

plt.gcf().set_size_inches(9,7)
plt.title('United States, Active Cases')

america1, america2, america3 = d.iloc[:12], d.iloc[12:21], d.iloc[21:]

ExponentQuadraticModel(pd.concat([america2,america3])).plot(before=12, after=7, color='magenta')

p = plot_fact(america1, label='US 1')
ExponentLinearModel(america1).plot(after=22, color=p[0]._color)

p = plot_fact(america2, label='US 2')
ExponentLinearModel(america2).plot(before=13, after=15, color=p[0]._color)

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
    format_axes(yscale='linear')
plt.gcf().set_size_inches(8,10)
y_max = float('-inf')
for i, country in enumerate([
        'US','Italy','Germany'
        ,'Switzerland','Canada','China','Ukraine'
        , 'New Zealand','North Macedonia']):

    d = data_hopkins.query(
        f"Country == '{country}' & (-18 <= DayN <= 0)"
    ).groupby('DayN').Active.sum()

    p = plot_fact(d, label=country)
    ExponentLinearModel(d.iloc[-8:-2]).plot(after=20, color=p[0]._color)
    y_max = max(y_max, d.max())
format_axes()
plt.ylim( plt.gca().get_ylim()[0], y_max*2 )
plt.legend(loc='lower right');
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
plt.gcf().set_size_inches(8,4)
plt.gca().set_yscale('log')
d = data_hopkins.query(
    "Country == 'Ukraine' & (-8 <= DayN <= 0)"
).groupby('DayN').Active.sum()

p = plot_fact(d, label='Ukraine')
ExponentLinearModel(d.iloc[:-2]).plot(after=22, color=p[0]._color)
plt.hlines(3.5e+4, -9, 20, color = 'red', label='Available ICUs Suffice')
format_axes()
plt.grid(True, axis='x', which='both');
plot_country('Mainland China'
             , plot=(date2day('2020-01-28'),date2day('2020-02-12'))
             , fit=(None,-1)
             , forecast=(15,25)
             , model=ExponentQuadraticModel)
plot_country(['Mainland China','China']
             , plot=(date2day('2020-02-13'),0)
             ,fit=(1,-25)
             , forecast=(30,15)
             , model=ExponentQuadraticModel)

format_axes()
plt.gca().set_yscale('linear')
plt.legend(loc='upper right');
plot_country(['South Korea','Korea, South']
             , plot=(date2day('2020-02-13'),0)
             #,fit=(1,-10)
             , forecast=(30,15)
             , model=ExponentQuadraticModel)

format_axes()
#plt.xlim(date2day('2020-01-10'), date2day('2020-03-20'))
plt.gca().set_yscale('linear')
plot_country('US', plot=(date2day('2020-03-30'),0), forecast=(0,30)
             , model=ExponentQuadraticModel)
plt.grid(True)
format_axes('linear')
for states in [
        ['Florida','New York','Michigan','Texas'],
        ['California','North Carolina','Washington']
]:
    plt.gca().set_yscale('log')
    plt.gcf().set_size_inches(8,6)

    for state in states:
        plot_country('US', state, plot=(-10,0), fit=(None,-1), forecast=(0,21)
                 , model=ExponentQuadraticModel)
    format_axes()
    plt.grid(True, axis='x', which='both')
    plt.ylim(500, min(plt.gca().get_ylim()[1], 1e6))
    plt.legend(loc='lower left')
    plt.savefig('SelcetedStates.png',format='png')
    plt.show()
plt.gca().set_yscale('log')
plt.gcf().set_size_inches(8,3)
d = data_hopkins.query(
    "Country == 'Switzerland' & -14<=DayN"
).groupby('DayN').Active.sum()

p = plot_fact(d, label='Switzerland')
ExponentQuadraticModel(d.iloc[:-7]).plot(after=15, color=p[0]._color)

format_axes()
plt.ylim(1000, 20000);
plt.gca().set_yscale('log')
plt.gcf().set_size_inches(8,3)
plot_country('Germany', plot=(-14,-2), forecast=(0,21)
             , model=ExponentQuadraticModel)
format_axes()
plt.grid(True, axis='x', which='both');
#plt.hlines(3.5e+4, 0, 21, color = 'red', label='Available ICUs Suffice')
plt.gcf().set_size_inches(10,8)
plot_country('Ukraine', plot=(-15,0), fit=(int(date2day('2020-04-06')-1),None), forecast=(0,35)
             , model=ExponentQuadraticModel)

format_axes()
plt.grid(True, axis='x', which='major')
plt.ylim(400, 50000);
plt.savefig('Ukraine.png',format='png')
plt.gcf().set_size_inches(8,4)
plot_country('North Macedonia', plot=(-20,0), fit=(-7,None), forecast=(0,21)
             , model=ExponentQuadraticModel)
plt.ylim(10, 10000)
format_axes()
plt.grid(True, axis='x', which='both')
np.quantile([1,2,3,4,5,float('inf'),float('inf')], [0.5])
plt.gcf().set_size_inches(10,4)
plot_country('Canada', plot=(-14,0), fit=(None,None), forecast=(0,28)
             , model=ExponentQuadraticModel)
format_axes('linear')
plt.grid(True, axis='x', which='both')
plt.gca().set_yscale('log')
plt.gcf().set_size_inches(12,3)
d = data_hopkins.query(
    "-14 <= DayN"
).groupby('DayN').Active.sum()

p = plot_fact(d, label='World')
ExponentQuadraticModel(d).plot(after=60, color=p[0]._color)

format_axes('linear')
plt.grid(True, axis='x', which='both')
#!rm -rf COVID-19 tests-daily.csv