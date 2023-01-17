#Notebook with basic EDA for the new cases per countries and curve fitting exercise for basic prediction

#Using dataset from WHO and population data

#try to learn from countries that passed the infection peak



import os

import pandas as pd

import numpy as np

from scipy.interpolate import Rbf

from scipy.optimize import curve_fit



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



#Uploaded dataset from World Health Organization with cases per country (will update with dataset from Johns Hopkins)

#and population per country file from World Bank

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Population per country from World Bank at: https://data.worldbank.org/indicator/SP.POP.TOTL

df_pop = pd.read_excel(os.path.join('/kaggle/input','who-cases-dataset-and-wdi-country-population',

                                    'API_SP.POP.TOTL_DS2_en_excel_v2_887218.xls'),skip_rows=3,header=3)
df_pop.head()
#modify name for South Korea and Iran to match other databases

df_pop.loc[df_pop['Country Name']=='Korea, Rep.','Country Name'] = 'South Korea'

df_pop.loc[df_pop['Country Name']=='Iran, Islamic Rep.','Country Name'] = 'Iran'
#using 2018 data, latest available at World Bank as of now

df_pop = df_pop.loc[:,['Country Name','2018']].set_index('Country Name')
df_pop
df_pop.describe().astype('int64')
[s for s in df_pop.index.to_list() if (s.find('Korea')>-1)] #verify South Korea naming
#Data source World Health Organization compiled by Oxford University at: https://ourworldindata.org/coronavirus-source-data

df_WHO = pd.read_csv(os.path.join('/kaggle/input','who-cases-dataset-and-wdi-country-population',

                                  'WHO_full_data2003.csv'))

df_corona = df_WHO[df_WHO.location!='World'].copy() #we'll do the analysis by country
df_corona.head()
#replace day with 19K new cases caused by measure change in China with average of near dates

maxindx = df_corona.loc[df_corona.location=='China',:].new_cases.idxmax()

avg_smooth = (df_corona.new_cases[maxindx-1]+df_corona.new_cases[maxindx+1])/2

df_corona.loc[maxindx,'new_cases']=avg_smooth
#add population column using World Bank data

df_corona = df_corona.set_index('location')

df_corona['Population']=df_pop

df_corona['new_cases_per_pop']=df_corona['new_cases']/df_corona['Population']*100000

df_corona.reset_index(inplace=True)
#check number of data points available per country

df_corona.groupby('location').date.count().describe()
#check distribution of known total cases per country

df_corona.groupby('location').total_cases.max().describe()
len(df_corona.location.unique()) #number of countries in dataset
def filter_out(x):

    f1 = x['total_cases'].max() > 10      #at least 10 sick cases

    f2 = x['Population'].min()  > 1000000 #pop more than 1M, small countries with distorted percent of confirmed cases

    f3 = len(x) > 13                      #more than 2 weeks of data

    return f1 & f2 & f3
#create reduced filtered dataset for analysis

df_analysis = df_corona.groupby('location').filter(filter_out).copy()
len(df_analysis.location.unique()) #number of countries for analysis
df_analysis.location.unique()
df_analysis.info()
#for each country filter out all dates before there are at least 5 new daily cases

df_filt = pd.DataFrame(columns=df_analysis.columns)

for grp in df_analysis.groupby('location'):

    start_indx = grp[1].loc[grp[1].new_cases >= 5,:].index

    if len(start_indx) > 0:

        df_filt = pd.concat([df_filt,grp[1].loc[start_indx[0]:,:]])
#remove countries with less than 5 days of meaningful new daily cases

df_filt = df_filt.groupby('location').filter(lambda x: len(x.date)>4)
df_filt.groupby('location').date.count().describe()
ctry_lst = list(df_filt.location.unique())

len(ctry_lst) #final number of countries for analysis after filtering
def country_select(cntry,df):

    df_sel = df[df.location == cntry].reset_index(drop=True)

    #consistency check, fails for China after fixing 17/2 reports

    #assert(all(df_sel.total_cases == df_sel.new_cases.cumsum()))

    return df_sel.fillna(0)
#candidate functions definition for curve fitting

def exp_func(x,a,b):

    return a*np.exp(b*x)



def poly_func(x,a,b):

    return a*(x**b)
#loop over all countries

#check curve fit for exponential and polynomial function for countries which passed the peak of the infection

#uses new cases per day per 100K of the country population as time series values for radial basis function interpolation

#saves peak duration list for countries that seem to be after infection peak and the curve fitting params

peak_duration = {}

param_list = []

for ctry in ctry_lst:

    fig, ax = plt.subplots()

    df_country = country_select(ctry,df_filt)

    plt.plot(df_country.new_cases_per_pop,marker='o')

    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)

    plt.plot(rbf,marker='+')

    ax.set_title(ctry)

    plt.show()

    peak_indx = np.argmax(rbf)

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    if peak_reached:

        print("Peak reached in {0} days".format(peak_indx))

        peak_duration[ctry]=peak_indx

        #exponential fit check

        params1, cov_params1 = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],p0=[0.01,0.5])

        print("\nExponential fit parameters: a:{0:.2f} b:{1:.2f}".format(params1[0],params1[1]))

        print("Covariance matrix of parameters:")

        print(cov_params1)

        param_list.append(params1)

        #polynomial fit check

        params2, cov_params2 = curve_fit(poly_func,np.arange(peak_indx),rbf[:peak_indx],p0=[0.01,0.5])

        print("\nPolynomial fit parameters: a:{0:.2f} b:{1:.2f}".format(params2[0],params2[1]))

        print("Covariance matrix of parameters:")

        print(cov_params2)  

        #rough check for goodness of fit

        if np.diag(cov_params1).sum() > np.diag(cov_params2).sum():

            print("\nPolynomial fit seems better.")

        else:

            print("\nExponential fit seems better.")

    else:

        print("Peak not yet reached")
peak_duration
param_list #curve fitting result parameters
avg_duration = sum(peak_duration.values())/len(peak_duration)

avg_duration
max_duration = max(list(peak_duration.values()))

max_duration
#example rough predict based on average/max duration to peak using exponential fitting parameters

def predict_new_cases(ctry):

    df_country = country_select(ctry,df_filt)

    rbf=Rbf(df_country.index,df_country.new_cases_per_pop,smooth=5)(df_country.index)

    peak_indx = np.argmax(rbf)

    #are we at least 5 days after peak of new cases and peak detection 5 days after start?

    peak_reached = (len(df_country) > peak_indx+4) & (peak_indx > 5)

    if peak_reached:

        print("Peak reached in {0} days".format(peak_indx))

    else:

        print("Peak not yet reached")

    #exponential fitting

    params, _ = curve_fit(exp_func,np.arange(peak_indx),rbf[:peak_indx],p0=[0.01,0.5])

    #print("\nExponential fit parameters: a:{0:.2f} b:{1:.2f}".format(params[0],params[1]))

    #naive predict new cases

    if len(df_country) < int(np.ceil(avg_duration)):

        dur = int(np.ceil(avg_duration))

    elif len(df_country) < max_duration:

        dur = max_duration

    else:

        dur = len(df_country)+5

        print("Past due peaking, arbitrary 5 days prediction")

    for j in range(len(df_country)+1,dur+1):

        rbf = np.append(rbf,params[0]*np.exp(params[1]*j))

    cases = rbf*int(df_pop.loc[ctry])/100000

    fig, ax = plt.subplots()

    plt.plot(np.arange(len(df_country)+1),cases[:(len(df_country)+1)],marker='+')

    plt.plot(np.arange(len(df_country),len(cases)),cases[len(df_country):],marker='o')

    ax.set_title(ctry+" predicted new daily cases")

    plt.show()

    print("Total predicted infected till peak: {0:d}".format(int(sum(cases))))
predict_new_cases('Israel')
predict_new_cases('United States')
fig, ax = plt.subplots()

df_filt.groupby('location').plot(y='new_cases',use_index=False,ax=ax,figsize = (10,6), marker='o',

                                 legend=False,title='New daily cases for each country')

ax.set_xlabel('Days from measurement start')

ax.set_ylabel('New daily cases')

plt.show()
#show maximum daily new cases as a percentage of population, "wall of shame" for country measures :-)

df_filt.groupby('location').new_cases_per_pop.max().sort_values(ascending=False)
df_filt = df_filt.set_index('location')

df_filt['max_ratio'] = df_filt.groupby('location').new_cases_per_pop.max()

df_filt.reset_index(inplace=True)

df_filt.sort_values(by=['max_ratio'],ascending=False,inplace=True,kind='mergesort') #mergesort is stable
#show graphs of daily new cases in batches of 5 to be able to see difference, sorted by severity

i = 0

for ctry,grp in df_filt.groupby('location',sort=False):

    if i%5==0: #new plot for group

        fig, ax = plt.subplots()

        ax.set_xlabel('Days from measurement start')

        ax.set_ylabel('New daily cases per 100K')

    grp.plot(y='new_cases_per_pop',use_index=False, ax=ax, figsize=(10,6), label=ctry, marker='o',

                                legend=True,title='New daily cases per 100K for each country')

    i +=1



plt.show()