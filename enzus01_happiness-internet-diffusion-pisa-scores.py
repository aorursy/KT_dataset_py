#Importing the fundamental librariers

%matplotlib inline

import pandas as pd

from pandas.plotting import scatter_matrix

from pandas.tools.plotting import scatter_matrix

import numpy as np

import matplotlib.pyplot as plt



#%matplotlib notebook



import warnings

warnings.filterwarnings('ignore')

PISA = pd.read_csv('../input/pisa-scores-2015/Pisa mean perfromance scores 2013 - 2015 Data.csv', nrows=1161)

PISA=PISA[['Country Name','Series Name','2015 [YR2015]']]

PISA.columns=['Country','Series','Score_2015']

PISA['Score_2015'][PISA['Score_2015']=='..']=np.NaN

PISA['Score_2015']=PISA['Score_2015'].astype(float)



PISA=PISA.pivot_table(index=['Country'],columns=['Series'],values=['Score_2015'])



names2=['mathematics','mathematics_Female','mathematics_Male','reading','reading_Female','reading_Male','science','science_Female','science_Male']

PISA.columns=names2

PISA=PISA.reset_index()

PISA.corr()
PISA['PISA_average_country']=(PISA['mathematics']+PISA['reading']+PISA['science'])/3.0

PISA['Delta_Male-Female']=PISA['mathematics_Male']-PISA['mathematics_Female']+PISA['reading_Male']-PISA['reading_Female']+PISA['science_Male']-PISA['science_Female']

PISA=PISA[['Country','PISA_average_country','Delta_Male-Female']]

PISA=PISA.dropna()

PISA.head()
# Define regression parameter 

a, b = np.polyfit(np.array(PISA['Delta_Male-Female']), np.array(PISA['PISA_average_country']), deg=1)

f = lambda x: a * x + b



plt.figure(figsize=(9,9))

plt.scatter(PISA['Delta_Male-Female'],PISA['PISA_average_country'],c='green')



x = np.array([min(PISA['Delta_Male-Female']),max(PISA['Delta_Male-Female'])])

plt.plot(x,f(x),lw=2, c="blue",label="Regression line")

plt.title("Scatterplot PISA average vs Delta MAle-Female for Country")

plt.xlabel('Delta Male vs Female')

plt.ylabel('PISA Average Per Country')

plt.show()

WordInds = pd.read_csv('../input/world-development-indicators/Indicators.csv')

#first selection of columns

WordInds=WordInds[['CountryName','IndicatorCode','Year','Value']]

#Filter for one year...

WordInds=WordInds[WordInds['Year']==2013]



#Select the indicators of interest

indicators=['IT.NET.USER.P2','NY.GDP.PCAP.CD','SE.ADT.LITR.FE.ZS','SE.ADT.LITR.MA.ZS','SE.ADT.LITR.ZS',

            'SG.VAW.REFU.ZS','SP.DYN.IMRT.IN']



WordInds=WordInds[WordInds['IndicatorCode'].isin(indicators)]

#expand the colum indicatorCode into a wide table..

WordInds=WordInds.pivot_table(index=['CountryName'],columns=['IndicatorCode'],values=['Value'])

#Rename the colummns

colonne=['InternetUsr','GDPPerCapita','AdultLitFem','AdultLiMal','AdultLit',

         'WomenWBelieve','MortaInfant']

WordInds.columns=colonne

#Creating a new feature based on difference onliteracy between male and female

WordInds['AdulDelta']=WordInds['AdultLiMal']-WordInds['AdultLitFem']



#Extract the Country name from the Index

WordInds=WordInds.reset_index()  

WordInds.head() 
_=scatter_matrix(WordInds, alpha=0.2, figsize=(10, 10), diagonal='kde')
Happiness = pd.read_csv('../input/world-happiness/2015.csv')

Happiness=Happiness[['Country','Happiness Score']]
Happiness.head()
#fix a couple of countries

Happiness['Country'][Happiness['Country']=='Russia']="Russian Federation"

Happiness['Country'][Happiness['Country']=='Macedonia']="Macedonia, FYR"
Global=pd.merge(WordInds,PISA, how='left', left_on='CountryName', right_on='Country')

Global=pd.merge(Global,Happiness, how='left', left_on='CountryName', right_on='Country')

Global.describe()
Global=Global[['CountryName', 'InternetUsr', 'GDPPerCapita', 

          'AdultLit', 'WomenWBelieve', 'MortaInfant', 'AdulDelta',

        'PISA_average_country', 'Delta_Male-Female', 

       'Happiness Score']]
sm=scatter_matrix(Global, alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.figure(figsize=(10,9))

cm = plt.cm.get_cmap('RdYlBu')

#xy = range(100)

sc = plt.scatter(Global['InternetUsr'],Global['MortaInfant'], c=Global['Happiness Score'], vmin=0, vmax=7, s=35, cmap=cm)

plt.colorbar(sc)

plt.scatter(Global['InternetUsr'],Global['WomenWBelieve'],c ='green',s=55)

plt.title('Internet vs Mortal Infant + Woman Belive is Correct to be Beaten by Husband)')

plt.ylabel('% Internet Users')

plt.xlabel('MortaInfant')

plt.Annotation('green dots are countries where Woman Belive is Correct to be Beaten by Husband',xy=(2, 1), xytext=(3, 1.5),

            arrowprops=dict(facecolor='black', shrink=0.05),)

plt.show()

GlobalB=Global.dropna()

a, b = np.polyfit(np.array(GlobalB['InternetUsr']),np.array(GlobalB['MortaInfant']) , deg=1)



f = lambda x: a * x + b



#cm = plt.cm.get_cmap('seismic')

cm = plt.cm.get_cmap('RdYlBu')



fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(11,11), sharex=False, sharey=False)

ax1.scatter(Global['MortaInfant'],Global['InternetUsr'], c=Global['Happiness Score'],s=35, vmin=0, vmax=8,  cmap=cm)

ax1.set_xlabel('Mortal Infants')



#ax1.scatter(GlobalB['MortaInfant'],GlobalB['WomenWBelieve'], c='Black',s=35)



ax1.set_ylabel('% Internet Users')

ax1.set_title('Internet User vs Mortal Infants and Happiness Index')



#x = np.array([0,100])

#ax1.plot(f(x),x,lw=2, c="blue",label="Regression line")



ax2.scatter(Global['PISA_average_country'],Global['Delta_Male-Female'], c=Global['Happiness Score'], vmin=0, vmax=8, s=35, cmap=cm)

ax2.set_xlabel('PISA average Per Country')

ax2.set_ylabel('Difference between Male vs Female')

ax2.set_title('PISA and Happiness Index')



a, b = np.polyfit(np.array(PISA['PISA_average_country']),np.array(PISA['Delta_Male-Female']), deg=1)

f = lambda x: a * x + b



x = np.array([min(PISA['PISA_average_country']),max(PISA['PISA_average_country'])])

ax2.plot(x,f(x),lw=2, c="blue",label="Regression line")



fig.colorbar(sc)