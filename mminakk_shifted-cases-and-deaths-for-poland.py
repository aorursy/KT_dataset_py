import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
plt.style.use('seaborn')
font = {'size'   : 40}
plt.rc('font', **font)

deaths_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv').groupby('Country/Region').sum().reset_index()
cases_raw = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv').groupby('Country/Region').sum().reset_index()
countries = ['China', 'Italy', 'US', 'Germany', 'Iran', 'Spain', 'France', 'Korea, South', 'United Kingdom','Poland']
colors = ['pink', 'black', 'green', 'brown', 'darkorchid', 'cyan', 'darkslategray', 'darkorange', 'b','red']

cases_raw.set_index('Country/Region', inplace=True)
deaths_raw.set_index('Country/Region', inplace=True)

cdeaths = deaths_raw.loc[countries,:].drop(columns=['Lat','Long'])
ccases  = cases_raw.loc[countries,:].drop(columns=['Lat','Long'])

deaths_shift = {}
cases_shift  = {}
nmax = 40

#starting point 
ctreshold = 100
dtreshold = 10


for country in countries:

  deaths_shift[country] = []
  cases_shift[country] = []

  for cval, dval in zip(ccases.loc[country],cdeaths.loc[country]):
    #print(value)
    if cval >= ctreshold:
      cases_shift[country].append(cval)

    if dval >= dtreshold:
      deaths_shift[country].append(dval)

  cases_shift[country] = cases_shift[country][:nmax]
  deaths_shift[country] = deaths_shift[country][:nmax]
def create_plot():
    fig, (ax1, ax2) = plt.subplots(2,figsize=(15,15))
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    #spagetti code for doubles every N-days
    T = np.arange(0, nmax)
    mlist = ['--',':','-.']
    for j,l in zip([2,4,7],[0,1,2]):

        T2 = []
        T3 = []
        X2 = []
        k = 0
        for i in T[::j]:
          X2.append(i)
          T2.append(ctreshold*2**(T)[k])
          T3.append(dtreshold*2**(T)[k])
          k += 1

        ax1.plot(X2,T2,mlist[l],label='Doubles every {} days'.format(j),color='grey')
        ax2.plot(X2,T3,mlist[l],label='Doubles every {} days'.format(j),color='grey')

    for country, color in zip(countries, colors):
      ax1.plot(cases_shift[country], label=country, color=color, marker='*')
      ax2.plot(deaths_shift[country], label=country, color=color, marker='*')

    ax1.set_ylabel('Number of Cases')
    ax2.set_ylabel('Number of Deaths')
    ax1.set_xlabel('Days starting at {} cases'.format(ctreshold))
    ax2.set_xlabel('Days starting at {} deaths'.format(dtreshold))

    
    ax1.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    ax2.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    return fig
create_plot()

plt.tight_layout()
plt.show()

#    ax2.tight_layout()

plt.savefig('./shifted_cases_and_deaths_for_POLAND.png')



# display animation
from IPython.display import Image
Image(filename="shifted_cases_and_deaths_for_POLAND.png")



