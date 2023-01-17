# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/covid19-community-mobility-dataset/world_mobility_with_covid_infection_count.csv')

# Any results you write to the current directory are saved as output.
dx=df#backup dataset

dx.columns
#comparing india v/s world mobility

df=df[(df['COUNTRY_REGION']=='India')]#load india dataset

plt.style.use('seaborn-darkgrid')

my_dpi=96

plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)

dx.dropna()

x=pd.DataFrame(dx.dtypes)

lst=[]

for r in x.itertuples():

    if r[1]!='object':

        lst.append(r[0])

dx.dropna()

collist=['RETAIL_AND_RECREATION_PCT','GROCERY_AND_PHARMACY_PCT','PARKS_PCT']

#average on world data

dfavg=dx.groupby(['DAY_CT']).mean()

dfavg.reset_index(inplace=True)
plt.rcParams["figure.figsize"] = (15,8)

#GROCERY_AND_PHARMACY_PCT

plt.plot(dfavg['DAY_CT'], dfavg['GROCERY_AND_PHARMACY_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['GROCERY_AND_PHARMACY_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.GROCERY_AND_PHARMACY_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.GROCERY_AND_PHARMACY_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on GROCERY AND PHARMACY_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("GROCERY AND PHARMACY_PCT Percentage")

#WORKPLACES_PCT

plt.plot(dfavg['DAY_CT'], dfavg['WORKPLACES_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['WORKPLACES_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.WORKPLACES_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.WORKPLACES_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on WORKPLACES_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#PARKS_PCT

plt.plot(dfavg['DAY_CT'], dfavg['PARKS_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['PARKS_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.PARKS_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.PARKS_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on PARKS_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#RETAIL_AND_RECREATION_PCT

plt.plot(dfavg['DAY_CT'], dfavg['RETAIL_AND_RECREATION_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['RETAIL_AND_RECREATION_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.RETAIL_AND_RECREATION_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.RETAIL_AND_RECREATION_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on RETAIL_AND_RECREATION_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#TRANSIT_STATIONS_PCT

plt.plot(dfavg['DAY_CT'], dfavg['TRANSIT_STATIONS_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['TRANSIT_STATIONS_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.TRANSIT_STATIONS_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.TRANSIT_STATIONS_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on TRANSIT_STATIONS_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#RESIDENTIAL_PCT

plt.plot(dfavg['DAY_CT'], dfavg['RESIDENTIAL_PCT'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.plot(df['DAY_CT'], df['RESIDENTIAL_PCT'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df.RESIDENTIAL_PCT.tail(1),'India',horizontalalignment='left',size='large' ,color='Blue')

plt.text(57.2, dfavg.RESIDENTIAL_PCT.tail(1),'World',horizontalalignment='left',size='large' ,color='Blue')

plt.title("Evolution of India vs Rest of the world on RESIDENTIAL_PCT Percentage", loc='left', fontsize=12, fontweight=0, color='blue')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#all in one #LockDown India beat the world

lst_params=['RETAIL_AND_RECREATION_PCT','GROCERY_AND_PHARMACY_PCT','PARKS_PCT','TRANSIT_STATIONS_PCT','WORKPLACES_PCT','RESIDENTIAL_PCT']

for x in lst_params:

    plt.plot(dfavg['DAY_CT'], dfavg[x], marker='', color='orange', linewidth=3, alpha=0.6)

    plt.text(57.7, dfavg[x].tail(1),'World '+x,horizontalalignment='left',size='large' ,color='Orange')

    plt.plot(df['DAY_CT'], df[x], marker='', color='red', linewidth=3, alpha=0.7)

    plt.text(57.2, df[x].tail(1),'India '+x,horizontalalignment='left',size='large' ,color='Red')

plt.xlabel("Day Count")

plt.ylabel("Percentage")    
#all in one #LockDown India beat the world

lst_params=['INC_RETAIL_AND_RECREATION_PCT','INC_GROCERY_AND_PHARMACY_PCT','INC_PARKS_PCT','INC_TRANSIT_STATIONS_PCT','INC_WORKPLACES_PCT','INC_RESIDENTIAL_PCT']

for x in lst_params:

    plt.plot(dfavg['DAY_CT'], dfavg[x], marker='', color='orange', linewidth=3, alpha=0.6)

    plt.plot(df['DAY_CT'], df[x], marker='', color='red', linewidth=3, alpha=0.7)

plt.xlabel("Day Count")

plt.ylabel("Percentage")    
#Covid-19 increase rate 

x='COVID_COUNTRY_INC_RATE'

plt.plot(dfavg['DAY_CT'], dfavg['COVID_COUNTRY_INC_RATE'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.text(57.7, dfavg['COVID_COUNTRY_INC_RATE'].tail(1),'World',horizontalalignment='left',size='large' ,color='Orange')

plt.plot(df['DAY_CT'], df['COVID_WORLD_INC_RATE'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df['COVID_WORLD_INC_RATE'].tail(1),'India',horizontalalignment='left',size='large' ,color='Red')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
    #COVID_CNTY_NEW_CASES

x='COVID_CNTY_NEW_CASES'

plt.plot(dfavg['DAY_CT'], dfavg['COVID_CNTY_NEW_CASES'], marker='', color='orange', linewidth=3, alpha=0.7)

plt.text(57.7, dfavg['COVID_CNTY_NEW_CASES'].tail(1),'World',horizontalalignment='left',size='large' ,color='Orange')

plt.plot(df['DAY_CT'], df['COVID_CNTY_NEW_CASES'], marker='', color='red', linewidth=3, alpha=0.7)

plt.text(57.2, df['COVID_CNTY_NEW_CASES'].tail(1),'India',horizontalalignment='left',size='large' ,color='Red')

plt.xlabel("Day Count")

plt.ylabel("Percentage")
#removing ouliers for seaborn charts regplot with 1000 bins on COVID_CNTY_NEW_CASES

Q1 = df.quantile(0.15)

Q3 = df.quantile(0.85)

IQR = Q3 - Q1

dfOutliners=((dx < (Q1 - 1.5 * IQR)) | (dx > (Q3 + 1.5 * IQR))).sum()

dx = dx[~((dx < (Q1 - 1.5 * IQR)) |(dx > (Q3 + 1.5 * IQR))).any(axis=1)]

sns.regplot(x=dx["WORKPLACES_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="WORKPLACES_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#jointplot

jplot=sns.jointplot(x='TRANSIT_STATIONS_PCT', y='COVID_CNTY_NEW_CASES', data=dx)
sns.regplot(x=dx["WORKPLACES_PCT"], y=dx["COVID_CNTY_NEW_CASES"],line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="WORKPLACES_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#RETAIL_AND_RECREATION_PCT

sns.regplot(x=dx["RETAIL_AND_RECREATION_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="RETAIL_AND_RECREATION_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#GROCERY_AND_PHARMACY_PCT

sns.regplot(x=dx["GROCERY_AND_PHARMACY_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="GROCERY_AND_PHARMACY_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#PARKS_PCT

sns.regplot(x=dx["PARKS_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="PARKS_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#TRANSIT_STATIONS_PCT

sns.regplot(x=dx["TRANSIT_STATIONS_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="TRANSIT_STATIONS_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#RESIDENTIAL_PCT

sns.regplot(x=dx["RESIDENTIAL_PCT"], y=dx["COVID_CNTY_NEW_CASES"],x_bins=1000, line_kws={"color":"r","alpha":0.7,"lw":5})

ax = sns.lineplot(x="RESIDENTIAL_PCT", y="COVID_CNTY_NEW_CASES",  markers=True,data=dx)
#Infection Bucket Creation

import pandas as pd

dfck=pd.read_csv('/kaggle/input/covid19-community-mobility-dataset/world_mobility_with_covid_infection_count.csv')

dfctry=dfck.groupby(['COUNTRY_REGION']).max()

dfctry.reset_index(inplace=True)

dfcctrycode=dfctry['COUNTRY_REGION']

xDF=pd.DataFrame()

for x in dfcctrycode:

    dfckfltr=dfck[dfck['COUNTRY_REGION']==x].sort_values(by=['DAY_CT'])

    dfckfltrday=dfckfltr[['DAY_CT','COVID_CONFIRMED','COVID_CNTY_NEW_CASES','COUNTRY_REGION']]

    lstVal=0

    for y in dfckfltrday.itertuples():

        if y[1]!=1:

            if lstVal*2 <= y[2] and y[2]!=0 and y[3]!=0:

                xDFtmp=dfck[(dfck['COUNTRY_REGION']==y[4]) & (dfck['DAY_CT']==y[1])]

                if len(xDF)==0:

                    xDF=xDFtmp

                else:

                    xDF=pd.concat([xDF,xDFtmp],ignore_index=True)

                lstVal=y[2]

        else:

            lstVal=0

#calculating no of days difference on doubling  the infection spread            

xDF['Double_Infection_Days'] = xDF['DAY_CT']-xDF.groupby(['COUNTRY_REGION'])['DAY_CT'].shift(1)

xDF['infection_bucket'] = xDF.groupby(['COUNTRY_REGION']).cumcount()
#grtting Indian and world top three covid-19 infected countries data

xDFi=xDF[(xDF['COUNTRY_REGION']=='India') | (xDF['COUNTRY_REGION_CODE']=='US')|(xDF['COUNTRY_REGION']=='Spain')|(xDF['COUNTRY_REGION']=='Italy')]

#getting Indian data

xDFir=xDFi[['COUNTRY_REGION','DATE_VAL','DAY_CT','Double_Infection_Days','COVID_CONFIRMED','infection_bucket']][xDFi['COUNTRY_REGION']=='India']

xDFir
x = sns.lineplot(x="infection_bucket", y="Double_Infection_Days",  markers=True,data=xDFir,hue='COUNTRY_REGION')
ax = sns.lineplot(x="infection_bucket", y="Double_Infection_Days",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#RETAIL_AND_RECREATION_PCT

ax = sns.lineplot(x="infection_bucket", y="RETAIL_AND_RECREATION_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#GROCERY_AND_PHARMACY_PCT

ax = sns.lineplot(x="infection_bucket", y="GROCERY_AND_PHARMACY_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#PARKS_PCT

ax = sns.lineplot(x="infection_bucket", y="PARKS_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#TRANSIT_STATIONS_PCT

ax = sns.lineplot(x="infection_bucket", y="TRANSIT_STATIONS_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#WORKPLACES_PCT

ax = sns.lineplot(x="infection_bucket", y="WORKPLACES_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
#RESIDENTIAL_PCT

ax = sns.lineplot(x="infection_bucket", y="RESIDENTIAL_PCT",

                  hue="COUNTRY_REGION", style="COUNTRY_REGION",

                  markers=True, dashes=False, data=xDFi)
ax = sns.relplot(x="infection_bucket", y="RETAIL_AND_RECREATION_PCT",

                  col="COUNTRY_REGION",hue='COUNTRY_REGION',

                  markers=True, dashes=False, data=xDFi)
#Corr

coffDf=dx.corr()

coffDf=coffDf[['DAY_CT']]

coffDf.rename(columns={"DAY_CT": "WORLD"},inplace=True)

country_list=['Spain','Italy','United States','India']

for xntry in country_list:

    xfilter=dx[(dx['COUNTRY_REGION']==xntry)]

    xfilter=xfilter.corr()

    xcorrctry=xfilter[['DAY_CT']]

    xcorrctry.rename(columns={"DAY_CT": xntry},inplace=True)

    coffDf=pd.concat([coffDf,xcorrctry],axis=1)

coffDf
d=sns.heatmap(coffDf)
coffDf.plot.bar(title="Corr in World");