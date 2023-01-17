# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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

from pathlib import Path

import os

from tqdm.notebook import tqdm

from scipy.integrate import solve_ivp

import numpy

import datetime

from datetime import timedelta

import math

%matplotlib inline



import csv

import sys

import os

import subprocess

import scipy as sp

from scipy.io import savemat



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd





# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()



# read in the various datasets thankfully created here on Kaggle by Kagglers !

# first is the weather (temperatures , humidities, windfall etc in capitals of 154 countries from late january thru. late March) and Covid-19 confirmed cases.deaths and recoveries

train = pd.read_csv('../input/covidglobalforecastwk4/train.csv')

fullWederCaseData = pd.read_csv('../input/covid19-global-weather-data/temperature_dataframe.csv')

# food supply data per country - brokemn down into oercentages across various categories like meat,veg. sugar, alcohol etc.

diet_kcal = pd.read_csv('../input/covid19-healthy-diet-dataset/Food_Supply_kcal_Data.csv')

diet_quant = pd.read_csv('../input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv')

protein_quant = pd.read_csv('../input/covid19-healthy-diet-dataset/Protein_Supply_Quantity_Data.csv')

diet_descripts = pd.read_csv('../input/covid19-healthy-diet-dataset/Supply_Food_Data_Descriptions.csv')
fullWederCaseData.head(n=5)
fullWederCaseData.describe()

#fullWederCaseData.shape
# Check no. if NaaNs exist and if so remove and save to a new aseparate DF :

# drop province column and store this augemted Df as the province col gives misleading NaN:

fullWederCaseData_npprov = fullWederCaseData.drop(['province'],axis=1)

#fullWederCaseData_npprov

print(fullWederCaseData_npprov.isnull().sum().sum())
# where r these nans ? :

#pd.set_option('display.max_rows',100000);

#np.where(np.array(fullWederCaseData_npprov) == 'NaN')

fullWederCaseData_npprov.isna().any()

fullWederCaseData_npprov.shape
noNaNs1_fullWederCaseData = fullWederCaseData_npprov.dropna();

noNaNs1_fullWederCaseData.shape

# confirm no more NaN rows left:

print(noNaNs1_fullWederCaseData.isnull().sum().sum())
#compute correlations of all columns in dataframe and display correlation heatmap !:

def plot_correlation_map( df ):

    corr = noNaNs1_fullWederCaseData.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    );

plot_correlation_map( noNaNs1_fullWederCaseData )
noNaNs1_fullWederCaseData['deathRate'] = 0.0;

#fullWederCaseData['deathRate'] = 0.0



noNaNs1_fullWederCaseData['deathRate'] = np.where((noNaNs1_fullWederCaseData['cases'] != 0.0), noNaNs1_fullWederCaseData['fatalities']/noNaNs1_fullWederCaseData['cases'], 0.0)
# potentially - it is better to use the (tempperature independent measure of humidity - 'Absolute Humidity'), acc. to a paper from MIT - link ??

# this is more important and appropriate



# absolute humidity as a fucntion of REALATIVE HUMID. AND tEMMP. (IN degrees Kelvin), is given by the relation:



# Abs_H =  [Rho (saturation water vapor) * Rel_H] / (100 * R_nu * T)  ;    where R_nu is the specific gas constant for water vapour.And T is temperature.



# the water vapor saturation -  Rho can be found with thr Clausisus-Calpyron equation as :



##   Rho = 6.11 * Exp { 53.49 - (6808/T) - 5.09 * LN(T)}



# putting all of tius together we have the function for Absolute hiiumidity  Abs_H :



#            Abs_H =  [6.11 * Exp { 53.49 - (6808/T) - 5.09 * LN(T)} * Rel_H] / (100 * R_nu * T) 

    

#    All in terms of known quantities in te dataset per country - i.e rel_humidity and temperature. (celcius to kelvin conversion is just adding      

#    273.15 )

    

#define consts 

R_nu = 461.5



def celcius_to_Kelvin(T_c):

    

    T_K =  T_c + 273.15 #(6.11 * math.exp(53.49 - (6808/T) - 5.09 * math.log(T)) * Rel_H) / (100 * R_nu * T)

    

    return T_K



    

def Rel_Humid_to_AbsHumid(T,Rel_H):

    

    Abs_H =  (6.11 * math.exp(53.49 - (6808/celcius_to_Kelvin(T)) - 5.09 * math.log(celcius_to_Kelvin(T))) * Rel_H) / (100 * R_nu * celcius_to_Kelvin(T))

    

    return Abs_H



# run the above code to create a new Absolute humidity column for the data.

#type(fullWederCaseData)





#rel_h = fullWederCaseData['humidity']

noNaNs1_absHumid_fullWederCaseData = noNaNs1_fullWederCaseData;

#noNaNs1_absHumid_fullWederCaseData['abs_humidity'] = noNaNs1_fullWederCaseData.apply(lambda row: 6.11 * math.exp(53.49 - (6808/celcius_to_Kelvin(row.tempC)) - 5.09 * math.log(celcius_to_Kelvin(row.tempC))) * row.humidity / (0.001 * R_nu * celcius_to_Kelvin(row.tempC)),axis=1)



noNaNs1_absHumid_fullWederCaseData['abs_humidity'] = noNaNs1_fullWederCaseData.apply(lambda row: (6.112 * math.exp(17.67*row.tempC/(row.tempC + 243.15))*row.humidity*2.1674)/(row.tempC + 273.15),axis=1)



    

    #noNaNs1_absHumid_fullWederCaseData['abs_humidity'] = noNaNs1_fullWederCaseData.apply(lambda row: 6.11 * math.exp(53.49 - (6808/celcius_to_Kelvin(row.tempC)) - 5.09 * math.log(celcius_to_Kelvin(row.tempC))) * row.humidity / (0.001 * R_nu * celcius_to_Kelvin(row.tempC)),axis=1)
pd.set_option('display.max_rows',50)

noNaNs1_absHumid_fullWederCaseData.head(50)

#noNaNs1_absHumid_fullWederCaseData.shape
# now redo the correlation heatmap:

#compute correlations of all columns in dataframe and display correlation heatmap !:

def plot_correlation_map( df ):

    corr = noNaNs1_absHumid_fullWederCaseData.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    );

plot_correlation_map( noNaNs1_absHumid_fullWederCaseData )
## try group some abs_humidity ranges :

#pd.set_option('display.max_rows',10000)





noNaNs1_absHumid_fullWederCaseData

noNaNs1_absHumid_fullWederCaseData['sumRangeAbsHumid'] = noNaNs1_absHumid_fullWederCaseData.apply(lambda row: int(row.abs_humidity > 3.0 and row.abs_humidity < 9.0),axis=1) 

noNaNs1_absHumid_fullWederCaseData.head(100)
noNaNs1_absHumid_fullWederCaseData['sumRangeAbsHumid'].sum()

#noNaNs1_absHumid_fullWederCaseData['sumRangeAbsHumid'].shape
avgAbsHumidWetherPerCuntryDf = noNaNs1_absHumid_fullWederCaseData.groupby('country')['abs_humidity'].mean()

avgAbsHumidWetherPerCuntryDf.head(50)
maxCasesWetherPerCuntryDf = noNaNs1_absHumid_fullWederCaseData.groupby('country')['cases'].sum()

maxCasesWetherPerCuntryDf.head(200)
avgAbsHumidWetherPerCuntryDf_dFrame = avgAbsHumidWetherPerCuntryDf.to_frame().join(maxCasesWetherPerCuntryDf)
#for col in avgAbsHumidWetherPerCuntryDf_dFrame.columns: 

#    print(col)



fullIntervalHumidity = avgAbsHumidWetherPerCuntryDf_dFrame.cases.sum() # = avgAbsHumidWetherPerCuntryDf.to_frame().join(maxCasesWetherPerCuntryDf)   
high_HUmidCases = avgAbsHumidWetherPerCuntryDf_dFrame[avgAbsHumidWetherPerCuntryDf_dFrame.abs_humidity > 9.0 ]

highIntervalHumidity = high_HUmidCases.cases.sum()

#high_HUmidCases
diet_kcal.head(250)
diet_kcal.describe()
# be helpful to get detail on exacty what each food categories are 



diet_descripts
# clean the NA values - drop the rows containing these as a first basic 'cleaning'

diet_kcal=diet_kcal.dropna()

diet_kcal.head(1000)
# lowest confirmed covid-19 cases as a % of popultaion

numpy_diet_kcal = diet_kcal['Confirmed'].values.argsort();

diet_kcal.iloc[numpy_diet_kcal]
# highest confirmed covid-19 cases as % of popultaion

diet_kcal['Country'][np.array(diet_kcal['Confirmed']).argsort()[170-25:170]]
# create a new country- humidity only dataframe - for merging with the food one next

avgHumidWetherDf = fullWederCaseData.groupby('country')['humidity'].mean()

avgHumidWetherDf.head(50)

#avgHumidWetherDf.shape

#type(avgHumidWetherDf)
df1 = pd.DataFrame(data=avgHumidWetherDf.index, columns=['country'])

df2 = pd.DataFrame(data=avgHumidWetherDf.values, columns=['humidity'])

df = pd.merge(df1, df2, left_index=True, right_index=True)



#avgHumidWetherDf_DF['cuntries'] = fullWederCaseData['country']

#avgHumidWetherDf_DF

foodNHumidity_df = df;

foodNHumidity_df['nation'] = foodNHumidity_df['country']

foodNHumidity_df.drop(['country'],axis=1)
# create new humidity column from avg HUmid new Col created above with the weather df.

#and add this col onto the food col :

avgHumidWetherDf_df = pd.DataFrame(avgHumidWetherDf)

#df_wethernFood = diet_kcal.join(avgHumidWetherDf.set_index('country'), on='Country')

#avgHumidWetherDf_df



df_wethernFood = diet_kcal.join(foodNHumidity_df.set_index('nation'), on='Country')
# drop it's NaNs:

df_wethernFood.dropna()

# in deciding which type of correlation to use to examine deth rates  with cereal consumtion ,  - i.e pearson or spearman (linear or non-linear relationships) just look - plot how one varies with the other, 

#specifically death rate with cereal consumption per country.

#plt.scatter(diet_kcal['Cereals - Excluding Beer'],diet_kcal['Deaths'])
def plot_correlation_map( df ):

    corr = diet_kcal.corr()

    _ , ax = plt.subplots( figsize =( 12 , 18 ) )

    cmap = sns.diverging_palette( 360 , 16 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : 1.2 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 6 }

    );

plot_correlation_map( diet_kcal )




# try fit a regression to the clear apparent protection effect of cereal consumption to no. of confirmed cases:



import statsmodels.api as sm;



X = diet_kcal[['Cereals - Excluding Beer']];

Y = diet_kcal[['Confirmed']];

X  = sm.add_constant(X)



model  = sm.OLS(Y, X).fit()

predictions = model.predict(X) 



print_model = model.summary()

print(print_model)
## 'cereals -Excluding beer' category's actual description entails foodstuffs like oats wholegrain things like wheat rye and... their products such as wholegrain pastasa bread etc.

## Also includes of course many breakfast type cereals people would have with milk - eg porridge (oats) , wheat based brekki cereals likw all bran . etc. In the literature one of the main and best 

#suppported heath benefits of whilwgrain foods is against colorectal cancer. (controlled studies). Also good enough evidence to stringly suggest protection against heart disease 0 which is know straight

#from covid data as a bag comorbidity to have in terms of dying from the covid infection. Although this can explain more deaths , it does not explai the correlation with confirmed case numbers.



# A potential confounder would have to also be separTELy negatively crorelated with cases. i.e ''Oilcrops' category - could assume a health conscious person would also consume seeds and nuts with their

#cereals here or in parallel moreso than no helath conscious. Add tis ito a mutilpe regression model and check theeffect or confoundinig on the ceral correlation :



X = diet_kcal[['Cereals - Excluding Beer','Oilcrops']];

Y = diet_kcal[['Confirmed']];

X  = sm.add_constant(X)



model_2  = sm.OLS(Y, X).fit()

predictions = model_2.predict(X) 



print_model = model_2.summary()

print(print_model)



# try fit a regression to the clear apparent negative effect of meat & dairy consumption to the fatality rate:



import statsmodels.api as sm;



X = diet_kcal[['Animal Products']];

Y = diet_kcal[['Deaths']];

X  = sm.add_constant(X)



model  = sm.OLS(Y, X).fit()

predictions = model.predict(X) 



print_model = model.summary()

print(print_model)


X = diet_kcal[['Animal Products','Obesity']];

Y = diet_kcal[['Deaths']];

X  = sm.add_constant(X)



model_2  = sm.OLS(Y, X).fit()

predictions = model_2.predict(X) 



print_model = model_2.summary()

print(print_model)



def plot_correlation_map( df ):

    corr1 = diet_kcal.corr(method='spearman')

    _ , ax = plt.subplots( figsize =( 12 , 18 ) )   

    cmap = sns.diverging_palette( 360 , 16 , as_cmap = True )

    _ = sns.heatmap(

        corr1, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : 1.2 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 6 }

    );

plot_correlation_map( diet_kcal )
def plot_correlation_map( df ):

    corr2 = df_wethernFood.corr(method='pearson')

    _ , ax = plt.subplots( figsize =( 12 , 18 ) )

    cmap = sns.diverging_palette( 360 , 16 , as_cmap = True )

    _ = sns.heatmap(

        corr2, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : 1.2 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 6 }

    );

plot_correlation_map( df_wethernFood )


