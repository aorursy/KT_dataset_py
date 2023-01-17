#code used from COVID-19: Digging a bit deeper

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import math



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pylab import * 



from statsmodels.tsa.ar_model import AR

from sklearn.linear_model import LinearRegression



from scipy import integrate, optimize

from scipy.optimize import curve_fit

from matplotlib.pyplot import *



from colorama import Fore



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()





from pathlib import Path

data_dir = Path('./input/')



import os

#os.listdir(data_dir)
cleaned_data = pd.read_csv('../input/us-covid19-data/complete_data_new_format_US.csv', parse_dates=['Date'])

cleaned_data.head()

#print(cleaned_data)
print("External Data")

print(f"Earliest Entry: {cleaned_data['Date'].min()}")

print(f"Last Entry:     {cleaned_data['Date'].max()}")

print(f"Total Days:     {cleaned_data['Date'].max() - cleaned_data['Date'].min()}")
cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province_State':'country',

                     'Country_Region':'state',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



'''

    we have swapped country for state due to the difference in US Columns in JH data



'''





# cases 

#cases = ['confirmed', 'deaths', 'recovered', 'active']

cases = ['confirmed', 'deaths']





# Active Case = confirmed - deaths - recovered

#cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']



# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data

pop_italy = 60486683.

pop_skorea = 51000000.

pop_spain = 46749696.

pop_france = 65273511

pop_iran = 83992949.

pop_US = 331002651.

pop_UK = 67784927.

pop_japan = 127000000.

pop_germany = 83000000.

pop_mexico = 129000000.

pop_singapore = 5837230.


plot_titles = ['New York','New Jersey', 'Michigan', 'California' , 'Louisiana', 'Florida','Massachusetts','Pennsylvania','Illinois','Texas','Colorado','Arizona','Nevada',  'Washington']
train = data

xdata_range=[]

for i in range(0, 100, 1):

    xdata_range.append(i)
country_max=[]

country_inflect = []

country_c = []

deaths=[]





for k in range(0,len(plot_titles),1):



    confirmed_total_date_country = train[train['country']==plot_titles[k]].groupby(['date']).agg({'confirmed':['sum']})

    fatalities_total_date_country = train[train['country']==plot_titles[k]].groupby(['date']).agg({'deaths':['sum']})

    total_date_country = confirmed_total_date_country.join(fatalities_total_date_country)



    grouped_country = data[data['country'] == plot_titles[k]].reset_index()

    death = grouped_country.deaths

    conf  = grouped_country.confirmed

    date = grouped_country.date

    # Estimated Death Ratio for each country

    m,b = polyfit(conf, death, 1) 

    #print('Slope of Confirmed to Death ratio =', m)



    country_df = total_date_country[9:]

    country_df['day_count'] = list(range(1,len(country_df)+1))

    country_df



    ydata = [i for i in country_df.confirmed['sum'].values]

    xdata = country_df.day_count

    ydata = np.array(ydata, dtype=float)

    xdata = np.array(xdata, dtype=float)



    xdata_offset=[]

    ydata_offset=[]

    offset = 0



    def myFunc(days, InfPop, Inflection, c):

        y = (InfPop/(1+np.exp(-(days-Inflection)/c)))      

        return y



    for i in range(offset, len(xdata-offset), 1):

        xdata_offset.append(xdata[i]-offset)

        ydata_offset.append(ydata[i]-offset)

        

    x0 = np.array(xdata_offset, dtype=float)   

    y0 = np.array(ydata_offset, dtype=float)

    

    #fit the data, return the best fit parameters and the covariance matrix

    popt, pcov = curve_fit(myFunc, x0, y0)

    deaths.append(m*popt[0])

    #print('Country =', plot_titles[k] )



    

   

    '''

      Actual Data 

    '''



    #ydata_offset = np.hstack((ydata_offset, np.zeros(30) + np.nan))     

    x = np.array(xdata_range, dtype=float)

    y = np.hstack((y0, np.zeros(100-len(xdata)) + 0))     

   

    plt.figure(1,figsize=(10, 6))

    plt.plot(x, y, "g*" , label="Actual Confirmed-Cases")

    plt.plot(x, y*m, "ks", label="Deaths")



    #Calculate rate of change in Confirmed Cases

    ydiff0 = np.diff(y)

    ydiff  = np.hstack((0, ydiff0*10))

    plt.bar(x, ydiff, align='center', alpha=.6, color='green', label='Actual Confirmed-Case Rate of change')



 

    '''

      These are the Sigmoid Model Plots

    '''

    #xdata_range=[]

    ymaxpred=[]

    ypred=[]



    for i in range(0, 100, 1):

        #xdata_range.append(i)

        ymaxpred.append(popt[0])

        ypred.append((popt[0]/(1+np.exp(-(xdata_range[i] - popt[1])/popt[2])))  )



    #ypred = np.array(ypred, dtype=float)

    ydiff0=np.diff(ypred)

    ydiff = np.hstack((0,ydiff0*10))





    plt.plot(xdata_range, myFunc(xdata_range, popt[0], popt[1], popt[2]), "rs", label='Sigmoid Model Estimates')

    plt.bar(xdata_range, ydiff,align='center', alpha=.5, color='red', label='Estimated Confirmed-Case Rate of change')

    plt.plot(xdata_range, ymaxpred, 'bd' , linewidth=1, label='Maximum Infected Population')

    plt.plot(popt[1],popt[0]/2,label='Inflection Point', marker='o', markerfacecolor='blue', markersize=12)

    plt.plot(xdata_range,myFunc(xdata_range, popt[0], popt[1], popt[2])*m, linewidth=3,label ='Death Estimate')

    plt.xlabel('Days')

    plt.ylabel('Confirmed-Cases or Deaths')

    plt.xlim(0,100)

    plt.ylim(0,max(myFunc(xdata_range, popt[0], popt[1], popt[2]))+5000)

    plt.grid(True)

    plt.title("Actual Confirmed-Cases vs. Sigmoid Model Estimates")

    plt.legend(loc='upper left')

    plt.annotate('Inflection Point', color='blue', xy=(popt[1],popt[0]/2),  xycoords='data',

                xytext=(0.2, 0.5), textcoords='axes fraction',

                arrowprops=dict(facecolor='blue', shrink=0.05),

                horizontalalignment='right', verticalalignment='top',

                )



    plt.annotate('Rate of Change (x10)', color='green', xy=(popt[1],popt[0]/3 ),  xycoords='data',

                xytext=(.25, .2), textcoords='axes fraction',

                )

   

    

    plt.show()



    print()

    print("Sigmoid Model Estimates for", plot_titles[k],":")

    print("      Max Confirmed-Cases =",'\t', round(popt[0],2))

    print("   Inflection Point(days) = ",'\t', round(popt[1],2))

    print('\t', '\t',"   \tc =" , '\t',round(popt[2],2))

    print( '\t','Estimated Deaths =','\t',round(deaths[k],2))

    print()

    print('Inflection Point is in days from January 22, 2020')





    country_max.append(popt[0])

    country_inflect.append(popt[1])

    country_c.append(popt[2])
print('----------------------------------------------------------------------------------------------------------')

print('Index', '\t','Country','Max_Est_Confirmed\t','Est_Deaths\t','Inflec_Point(days)\t','c')

print('----------------------------------------------------------------------------------------------------------')



for k in range(0,len(plot_titles),1):

    print(k,'\t', plot_titles[k][:5],'\t','\t',round(country_max[k],3),'\t',round(deaths[k],3),'\t','\t',round(country_inflect[k],1),'\t','\t',round(country_c[k],2))

    

print() 

print('Inflection Point is in days from January 22, 2020')
