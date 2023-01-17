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

import os

#import world_bank_data as wb





#Load datasets

path = '../input/novel-corona-virus-2019-dataset/'

con = pd.read_csv(path+'time_series_covid_19_confirmed.csv')

rec = pd.read_csv(path+'time_series_covid_19_recovered.csv')

dea = pd.read_csv(path+'time_series_covid_19_deaths.csv')
#use a custom mapping file

df_map = pd.read_csv('../input/coronavirus/countryMapping.csv')



#standardize countries

scon = con.merge(df_map, how='left', on = 'Country/Region')

srec = rec.merge(df_map, how='left', on = 'Country/Region')

sdea = dea.merge(df_map, how='left', on = 'Country/Region')



#check missing country code

scon.loc[scon['Country Code'].isna(), 'Country/Region'].unique()
#Stack and merge dataframes

def stack_df(df, field):

    tmp = df.iloc[:,4:]

    tmp = tmp.groupby('Country Code').sum()

    tmp =tmp.stack().reset_index()

    tmp.columns=['country', 'dates', field]

    return tmp



scon = stack_df(scon, 'confirmed')

srec = stack_df(srec, 'recovered')

sdea = stack_df(sdea, 'deaths')

df = scon.merge(srec, on=['country', 'dates']).merge(sdea, on=['country', 'dates'])

df.head()
df['dates'] = pd.to_datetime(df.dates)

df['actives'] = df.apply(lambda x: x.confirmed - x.deaths - x.recovered, axis = 1)

df['lethality'] = df.apply(lambda x: 100*x.deaths / x.confirmed if x.confirmed > 0 else 0.0, axis = 1)



'''Convert cumulative Confirmed to new cases'''

newCases=[]

for cc in df.country.unique():

    sel =  df.loc[df.country==cc].sort_values('dates')

    cumul = sel.confirmed.values

    dates = sel.dates.values

    newCases.extend([(cc, dates[0], cumul[0])] + [(cc, dates[ix+1], i- cumul[ix]) for ix,i in enumerate(cumul[1:])])

newCases = pd.DataFrame(newCases, columns = ['country','dates','new_cases'])

df = df.merge(newCases, on=['country','dates'])

df.head()



'''Calculate prevalence & incidence from world bank population data'''

#Deprecated because can't !pip install world_bank_data

# wb_pop = pd.DataFrame(wb.get_series('SP.POP.TOTL', date='2018', id_or_value='id',simplify_index=True)).reset_index()

# wb_pop = wb_pop.rename(columns={'Country':'Country Code', 'SP.POP.TOTL':'population'})

#Replaced by

dfp = pd.read_csv('../input/coronavirus/world.csv')

wb_pop=  dfp.loc[dfp['Series Code'] == 'SP.POP.TOTL',['Country Code','2018 [YR2018]']]

wb_pop['2018 [YR2018]'] = wb_pop['2018 [YR2018]'].apply(lambda x: eval(x) if x !='..' else np.nan)

wb_pop = wb_pop.rename(columns={'2018 [YR2018]': 'population'})



df = df.merge(wb_pop, left_on = 'country', right_on='Country Code')

del df['Country Code']

df['prevalence'] = df.apply(lambda x: round(10000*x.confirmed/x.population,5), axis = 1)

df['incidence'] = df.apply(lambda x: round(10000*x.new_cases/x.population,5), axis = 1)

df.dropna(inplace=True)

df.head()
'''Apply R script to calculate Reproduction Factor from Epiestim package'''

#df.to_csv('covid19_epi.csv', index=False)

#if os.system('Rscript get_R.R')!=0:

#    print('Error in R script: get_R')



"""ReproductionFactor.csv is obtained from the R script get_R.R provided here:

library(EpiEstim)



covid19 <- read.csv("~/Documents/deep_learning/coronavirus/covid19_epi.csv")

covid19$dates<-as.Date(covid19$dates)

firstpass<-TRUE

for (i in unique(covid19$country))

{

  sel <- subset(covid19, country==i)



  #reproduction number

  df=subset(sel, select=c(dates, new_cases))

  names(df)[2] <- "I"

  df$I<-replace(df$I, df$I<0, 0)

  res_parametric_si <- estimate_R(df, method="parametric_si", config = make_config(list(mean_si = 3.96, std_si = 4.75)))

  si_param<-res_parametric_si$SI.Moments

  si_dist<-res_parametric_si$si_distr

  R<-subset(res_parametric_si$R, select=c(`Mean(R)`, `Quantile.0.05(R)`, `Median(R)`, `Quantile.0.95(R)`))

  R$dates<-res_parametric_si$dates[(2:(length(res_parametric_si$dates)-6))]

  R$country<-i

  if(firstpass)

  {

    concat<-R

    firstpass<-FALSE

  }

  else

    concat<-rbind(concat,R)#c(concat,R)

}



path<-"~/Documents/deep_learning/coronavirus/"

write.csv(concat,paste(path,"ReproductionFactor.csv")) 

"""



dfr = pd.read_csv('../input/coronavirus/ReproductionFactor.csv')

del dfr['Unnamed: 0']

dfr.columns=['mean', 'Q5', 'med', 'Q95', 'dates','country']

dfr['dates'] = pd.to_datetime(dfr.dates)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



def plotCountries(feature, df, title, country = None):

    weeks = mdates.WeekdayLocator()

    dayMonth_fmt = mdates.DateFormatter('%m-%d')

    fig, ax = plt.subplots(figsize = (12,6))

    if not country is None:

        tmp = df.loc[df.country== country]

        g=sns.lineplot(x="dates", y="mean", data=tmp)

        g.fill_between(x="dates", y1="Q5", y2="Q95", data=tmp, alpha=0.2)

    else:

        g = sns.lineplot(x="dates", y=feature, hue="country",  data=df)

    g.set_title(title)  

    g.xaxis.set_major_locator(weeks)

    g.xaxis.set_major_formatter(dayMonth_fmt)

    plt.xticks(rotation=30) 

    



def plotRegion(countries):

    #subset of selected countries

    sel = df.loc[df.country.isin(countries)]

    rsel = dfr.loc[dfr.country.isin(countries)]

    #Common X axis

    dates = [np.datetime_as_string( i, unit='D') for i in sel.dates.values]

    #create multiple subplots on one layout

    fig = make_subplots(

        rows=3, cols=2,

        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],

        subplot_titles=("Number of infectious cases", "Number of cases per 10 000 capita", "Number of new cases per 10 000 capita", "Lethalithy in %","Reproduction Factor"))

    #plot features from sel

    for idx, feature in enumerate(['actives','prevalence','incidence','lethality']):

        for i in sel.country.unique():

            tmp = sel.loc[sel.country == i]

            fig.add_trace(go.Scatter(x=dates, y=tmp[feature].values, mode='lines', name=i, legendgroup=i, showlegend= not idx), row=idx//2+1, col =idx%2+1)

    #plot R factor from rsel

    for i in sel.country.unique():

            tmp = rsel.loc[rsel.country == i]

            fig.add_trace(go.Scatter(x=dates, y=tmp['mean'].values, mode='lines', name=i, legendgroup=i, showlegend= False), row=3, col =1)        

    return fig.update_layout(height=800, title_text="Descriptive epidemiology in Europe"), sel, rsel #height=900, width=1200, 



def getLastR(df):

    return df.loc[df.groupby('country').dates.idxmax()].sort_values('mean',ascending=False)



def getCorrLethality(df, countries):

    return pd.DataFrame([(i, df.loc[(df.country == i)&(df.prevalence>0.2)][['lethality','prevalence']].corr().values[0][1]) for i in countries],\

             columns = ['country', 'corr']).sort_values('corr', ascending =False).dropna()



def choropleth(df, scope, field, uppad=0):

    #get min max of the feature for color scaling

    scales = df[field].describe()

    #define map for Reproduction Factor

    fig = px.choropleth(df, locations='country', color=field,

                               locationmode='ISO-3',

                               color_continuous_scale="Viridis",

                               range_color=(scales['min'], scales['max']-uppad),

                               scope=scope,

                               labels={field:'Reproduction Factor'}

                               #,width = 1200, height = 600

                              )

    return fig.update_layout()#autosize=False, width=1200, height=900,margin={"r":0,"t":0,"l":0,"b":0}

countries=['FRA', 'ITA', 'ESP', 'DEU', 'SWE','SWZ', 'GBR', 'BEL','AUT','NLD']

fig, sel, rsel = plotRegion(countries)

fig.show()
getCorrLethality(sel, countries)
plotCountries('mean',rsel,'Reproduction factor in France','FRA')
tmp = getLastR(rsel)

choropleth(tmp, 'europe', 'mean',4)
countries=['CHN', 'KOR', 'JPN','MYS','THA','IRN','IND','AFG','IRK']

fig, sel, rsel = plotRegion(countries)

fig.show()
tmp = getLastR(rsel)

choropleth(tmp, 'asia', 'mean')
getCorrLethality(sel, countries)
countries=['USA','CAN','MEX']

fig,sel, rsel = plotRegion(countries)

fig.show()
getCorrLethality(sel, countries)
tmp = getLastR(rsel)

choropleth(tmp, 'north america', 'mean')
from scipy.optimize import minimize, curve_fit

import scipy.integrate as spi



def diff_eqs(INPUT,t):  

    Y=np.zeros((3))

    V = INPUT   

    Y[0] = -beta * V[0] * V[1]

    Y[1] = beta * V[0] * V[1] - gamma * V[1]

    Y[2] = gamma * V[1]

    return Y  



def SIR(x, p):

    initial, beta, gamma, = p

    res = spi.odeint(diff_eqs,initial,x)

    return res[:,1]



#We select the vector to be fitted by the model



sel = df.loc[(df.country == 'FRA') & (df.dates> '2020-01-23'), 'actives'].values

duration= sel.shape[0]

pop = 67e6

trueCaseRate = 100

S0=0.82  #initial p(succeptible)

I0= sel[0]*trueCaseRate/pop  #initial p(infectious)

beta = 4/14

gamma = 1/14

res1 = SIR(range(0,150), [[S0, I0,0.0],beta, gamma])    

plt.plot(res1, '-r', label='Infectious')

y=trueCaseRate*sel/(S0*pop)

plt.plot(y, '-b', label='Infectious')

print('Days before the climax: %i'%(np.argmax(res1)-len(y)))

print('Max number of infectious cases = %i'%(max(res1)*pop/trueCaseRate))

plt.show()
#select the start of outbreak

sel = df.loc[(df.country == 'FRA') & (df.dates> '2020-01-23'), 'actives'].values

duration= sel.shape[0]

pop = 67e6

trueCaseRate = 100

S0=0.96  #initial p(succeptible)

I0= sel[0]*trueCaseRate/pop  #initial p(infectious)

beta = 2.5/9

gamma = 1/9

res2 = SIR(range(0,150), [[S0, I0,0.0],beta, gamma])    

plt.plot(res1, '-r', label='Infectious')

y=trueCaseRate*sel/(S0*pop)

plt.plot(y, '-b', label='Infectious')

print('climaxday: %i'%(np.argmax(res1)-len(y)))

print('Max number of infectious cases = %i'%(max(res2)*pop/trueCaseRate))
#select the start of outbreak

sel = df.loc[(df.country == 'FRA') & (df.dates> '2020-01-23'), 'actives'].values

duration= sel.shape[0]

pop = 67e6

trueCaseRate = 70

S0=0.86  #initial p(succeptible)

I0= sel[0]*trueCaseRate/pop  #initial p(infectious)

beta = 3/10

gamma = 1/10

res3 = SIR(range(0,150), [[S0, I0,0.0],beta, gamma])    

plt.plot(res3, '-r', label='Infectious')

y=trueCaseRate*sel/(S0*pop)

plt.plot(y, '-b', label='Infectious')

print('climaxday: %i'%(np.argmax(res3)-len(y)))

print('Max number of infectious cases = %i'%(max(res3)*pop/trueCaseRate))
models = pd.concat([pd.DataFrame(range(0,150)), pd.DataFrame(res1), pd.DataFrame(res2), pd.DataFrame(res3)], axis=1)

models.columns = ['days','model 1', 'model 2', 'model 3']

models.set_index('days', inplace=True)

models = models.stack().reset_index()

models.columns=['day','type','value']

models.head()



px.line(models, x="day", y= "value", color="type")
df_map = pd.read_csv('countryMapping.csv')

scon = con.merge(df_map, how='left', on = 'Country/Region')

df_cc = pd.read_csv('countryCodes.csv')

m2 = scon.loc[scon['Country Code'].isna()].merge(df_cc ,how='left', left_on='Country/Region', right_on='Country Name')

m2.loc[m2['Country Code_y'].isna(), 'Country/Region'].unique()

m2['Country Code'] = m2.apply(lambda x: x['Country Code_y'] if type(x['Country Code_y'])!=float else x['Country Code_x'], axis =1 )

m2[['Country/Region','Country Code']].groupby('Country/Region').agg({'Country Code': 'first'}).reset_index().to_csv('countryMapping2.csv', index=False)

#Fill missing codes by hand
df_map2 = pd.read_csv('countryMapping2.csv')

ndf_map = pd.concat([df_map, df_map2])

ndf_map.to_csv('countryMapping.csv', index=False)

ndf_map.shape