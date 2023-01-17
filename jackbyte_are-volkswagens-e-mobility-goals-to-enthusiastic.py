# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# First dataframe is amount of car registrations per fuel type per year in Germany

# data taken from https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/Umwelt/n_umwelt_z.html

d = {'petrol':[1695972,2608767,1669927,1651637,1555241,1502784,1533726,1611389,1746308,1986488,]

,'diesel':[1361457,1168633,1221938,1495966,1486119,1403113,1452565,1538451,1539596,1336776]

,'LPG':[14175,11083,8154,4873,11465,6257,6234,4716,2990,4400]

,'CNG':[11896,10062,4982,6283,5215,7835,8194,5285,3240,3723]

,'electric':[36,162,541,2154,2956,6051,8522,12363,11410,25056]

,'Hybrid':[6464,8374,10661,12622,21438,26348,27435,33630,47996,84675]

,'Plugin Hybrid':[0,0,0,0,408,1385,4527,11101,13744,29436]

,'sum allover':[3090040,3807175,2916260,3173634,3082504,2952431,3036773,3206042,3351607,3441262]

}

gs = pd.DataFrame(d, index=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])





# Any results you write to the current directory are saved as output.
ax = gs.plot(figsize=(15,10))

ax.grid(color='grey')

ax.set_title("Car registrations in Germany from 2008 to 2017")
ax = gs.electric.plot(figsize=(15,10))

ax.grid(color='grey')

ax.set_title("Electric car registrations in Germany from 2008 to 2017")

ax.plot(gs.electric.cumsum())
gs['electric + Hybrid'] = gs.electric + gs.Hybrid

ax = gs['electric + Hybrid'].plot(figsize=(15,10))

ax.grid(color='grey')

ax.set_title("Electric + Hybrid car registrations in Germany from 2008 to 2017")

ax.plot(gs['electric + Hybrid'].cumsum())
# insert rows until the year 2030

df = pd.DataFrame(gs)

for i in range(2018,2031):

    df.loc[i] = [None for c in gs.columns]



# extrapolation stolen from here https://stackoverflow.com/questions/22491628/extrapolate-values-in-pandas-dataframe/35959909#35959909

from scipy.optimize import curve_fit



# Function to curve fit to the data

def func(x, a, b, c, d):

    return a * (x ** 3) + b * (x ** 2) + c * x + d



# Initial parameter guess, just to kick off the optimization

guess = (0.5, 0.5, 0.5, 0.5)



# Create copy of data to remove NaNs for curve fitting

fit_df = gs



# Place to store function parameters for each column

col_params = {}



# Curve fit each column

for col in fit_df.columns:

    # Get x & y

    x = fit_df.index.astype(float).values

    y = fit_df[col].values

    # Curve fit column and get curve parameters

    params = curve_fit(func, x, y, guess)

    # Store optimized parameters

    col_params[col] = params[0]



# Extrapolate each column

for col in df.columns:

    # Get the index values for NaNs in the column

    x = df[pd.isnull(df[col])].index.astype(float).values

    # Extrapolate those points with the fitted function

    df[col][x] = func(x, *col_params[col])
ax = df.electric.plot(figsize=(15,10))

ax.grid(color='grey')

ax.set_title("Electric car registrations in Germany from 2008 to 2017")

ax.plot(df.electric.cumsum(), label = 'electric acumulated')

ax.plot(df['electric + Hybrid'])

ax.plot(df['electric + Hybrid'].cumsum(), label = 'Electric + Hybrid acumulated')

ax.legend()
ax = df.loc[df.index > 2018].electric.plot(figsize=(15,10))

ax.grid(color='grey')

ax.set_title("Electric car registrations in Germany from 2008 to 2017")

ax.plot(df.loc[df.index > 2018].electric.cumsum(), label = 'electric acumulated')

ax.plot(df.loc[df.index > 2018]['electric + Hybrid'])

ax.plot(df.loc[df.index > 2018]['electric + Hybrid'].cumsum(), label = 'Electric + Hybrid acumulated')

ax.legend()
vw = {'VW sales':[1187000,1247000,1279000,1257000,1264000]}

dvw = pd.DataFrame(vw, index=[2013,2014,2015,2016,2017])

df['VW Sales'] = dvw

df['VW % market share germany'] = (df['VW Sales'] / df['sum allover'])*100

ax = df[pd.notnull(df['VW Sales'])]['VW % market share germany'].plot(figsize=(15,10))

ax.set_title('Volkswagen market share germany in % from 2013 to 2017')

ax.grid(color='grey')