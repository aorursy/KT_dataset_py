import csv

import pandas as pd

import numpy as np

import matplotlib.pylab as pylab

%matplotlib inline



from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

import datetime
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tab = pd.read_csv("../input/country-pop-data/country_pop_data.csv")

con = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")



con_US = pd.read_csv("../input/latest-us/latest_US.csv")

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 16,

        }
country_df = pd.read_csv("../input/country-data/country_pop_data1.csv")

italy = country_df[country_df["Country/Region"]=="Italy"]

spain = country_df[country_df["Country/Region"]=="Spain"]

hubei = country_df[country_df["Province/State"]=="Hubei"]

china = country_df[country_df["Country/Region"]=="China"]

not_hubei= china[china["Province/State"]!="Hubei"]

japan = country_df[country_df["Country/Region"]=="Japan"]

s_korea = country_df[country_df["Country/Region"]=="Korea, South"]

sing = country_df[country_df["Country/Region"]=="Singapore"]

iran = country_df[country_df["Country/Region"]=="Iran"]

us = country_df[country_df["Country/Region"]=="US"]

uk = country_df[country_df["Country/Region"]=="United Kingdom"]

ger = country_df[country_df["Country/Region"]=="Germany"]

# makes cumulative totals for each country

china = con.loc[con['Country/Region'] == 'China']

aus = con.loc[con['Country/Region'] == 'Australia']

can = con.loc[con['Country/Region'] == 'Canada']

total_china = 0

total_aus = 0

total_can = 0

for c in range(len(china)):

    last_china = china.iloc[c,-1]

    total_china = total_china + last_china

for b in range(len(aus)):

    last_aus = aus.iloc[b,-1]

    total_aus = total_aus + last_aus

for d in range(len(can)):

    last_can = can.iloc[d,-1]

    total_can = total_can + last_can    
#combines cases and country information with a row for each country 

table=con

table["Province/State"].fillna("*", inplace = True)

tab_latest = pd.DataFrame(columns=['SNo','Province/State','Country/Region','Confirmed','Confirmed per mill population','Date','% of Population'])

count = 3

for a in range(len(table)):

    state = table.iloc[a,0]

    country = table.iloc[a,1]

  

    last_conf = table.iloc[a,-1]

    last_date = table.columns.values[-1] 

    rows1 = tab.loc[tab['Country/Region'] == country]

    rows2 = rows1.loc[rows1['Province/State'] == state]

    if len(rows2.index.values) >0:

        count+=1

        pop = rows2.iloc[0,5]

        conf_m = last_conf/pop

        conf_mill = round(conf_m,2)

        conf_perc = conf_mill/10000

        tab_latest.loc[count] = [count,state,country,last_conf,conf_mill,last_date,conf_perc]

#combines cases and country information for Australia China and Canada and adds it to the dataframe above

tab_extra = pd.DataFrame(columns=['SNo','Province/State','Country/Region','Confirmed','Confirmed per mill population','Date','% of Population'])

countries = [ "Australia", "Canada","China"]

totals = [total_aus,total_can,total_china ]

count = 0

for each_country in countries:

    state = "*"    

    rows1 = tab.loc[tab['Country/Region'] == each_country]

    rows2 = rows1.loc[rows1['Province/State'] == state]   

    last_conf =totals[count]

    if len(rows2.index.values) >0:

        count +=1       

        pop = rows2.iloc[0,5]

        conf_m = last_conf/pop

        conf_mill = round(conf_m,2)

        conf_perc = conf_mill/10000

        

        tab_extra.loc[count] = [count,state,each_country,last_conf,conf_mill,last_date,round(conf_perc,2)]

frames = [tab_latest,tab_extra]

tab_comb = pd.concat(frames)
tab1 = tab_comb.loc[tab_comb["Confirmed"]>5000]

tab2 = tab1.sort_values(by ='Confirmed',ascending=False )

tab3 = tab2.loc[tab2['Province/State'] == '*']
date1 = str(last_date)

fig = plt.figure(figsize=(15,7))

plt.ylabel('Cases')

plt.xlabel('Countries with > 5000 cases')

plt.title('Reported Cases per Country (as of '+date1+')',fontdict=font)

 

plt.bar(tab3['Country/Region'],tab3['Confirmed'] )

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Reported Cases >5000 per Country")
fig = plt.figure(figsize=(15,7))

plt.ylabel('Cases (per million)')

plt.xlabel('Countries with > 1000 cases')

plt.title('Reported Cases per million of Population (as of '+date1+')',fontdict=font)

 

plt.bar(tab3['Country/Region'],tab3['Confirmed per mill population'] )

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Reported Cases per million of Population")
con_US2 = con_US.loc[con_US["Confirmed"]>1000]

con_US2_NY = con_US2.loc[con_US2["Province/State"]=="New York"]

con_US2_Cal = con_US2.loc[con_US2["Province/State"]=="California"]

con_US2_Mich = con_US2.loc[con_US2["Province/State"]=="Michigan"]

con_US2_NJ = con_US2.loc[con_US2["Province/State"]=="New Jersey"]

con_US2_ill = con_US2.loc[con_US2["Province/State"]=="Illinois"]

con_US2_flo = con_US2.loc[con_US2["Province/State"]=="Florida"]

con_US2_lou = con_US2.loc[con_US2["Province/State"]=="Louisiana"]

con_US2_pen = con_US2.loc[con_US2["Province/State"]=="Pennsylvania"]

con_US2_con = con_US2.loc[con_US2["Province/State"]=="Connecticut"]

con_US2_was = con_US2.loc[con_US2["Province/State"]=="Washington"]

con_US2_mas = con_US2.loc[con_US2["Province/State"]=="Massachusetts"]

con_US2_tex = con_US2.loc[con_US2["Province/State"]=="Texas"]

con_US2_ind = con_US2.loc[con_US2["Province/State"]=="Indiana"]

con_US2_nev = con_US2.loc[con_US2["Province/State"]=="Nevada"]

con_US2_ari = con_US2.loc[con_US2["Province/State"]=="Arizona"]

con_US2_wasDC = con_US2.loc[con_US2["Province/State"]=="District of Columbia"]

con_US2_mar = con_US2.loc[con_US2["Province/State"]=="Maryland"]

con_US2_geo = con_US2.loc[con_US2["Province/State"]=="Georgia"]

con_US2_mis = con_US2.loc[con_US2["Province/State"]=="Missouri"]

con_US2_wis = con_US2.loc[con_US2["Province/State"]=="Wisconsin"]

#added 18/4

con_US2_utah = con_US2.loc[con_US2["Province/State"]=="Utah"]

con_US2_vir = con_US2.loc[con_US2["Province/State"]=="Virginia"]

con_US2_col = con_US2.loc[con_US2["Province/State"]=="Colorado"]

con_US2_ohio = con_US2.loc[con_US2["Province/State"]=="Ohio"]

con_US2_sda = con_US2.loc[con_US2["Province/State"]=="South Dakota"]

con_US2_del = con_US2.loc[con_US2["Province/State"]=="Delaware"]

con_US2_rho = con_US2.loc[con_US2["Province/State"]=="Rhode Island"]

con_US2_nca = con_US2.loc[con_US2["Province/State"]=="North Carolina"]

con_US2_ken = con_US2.loc[con_US2["Province/State"]=="Kentucky"]

#con_US2.tail(9)
date_US = con_US.iloc[0,8]

my_colors = 'rgbkymc' 

figNY = plt.figure(figsize=(15,7))

plt.barh(con_US2_NY['Admin Area'],con_US2_NY['Confirmed'],color=my_colors)

plt.title("New York State")

#fig.suptitle('US Areas with reported cases > 1000 (as of '+date_US+')',fontdict=font)

print('Areas with more than 1000 cases (as of '+date_US+')')
#date_US = con_US.iloc[0,7]

my_colors = 'rgbkymc' 

#fig = plt.figure(figsize=(15,7))

#fig, axs = plt.subplots(1, 5, figsize=(25, 5))

#fig,((ax1,ax2),(ax3,ax4))= plt.subplots(2,2,figsize=(9,9))

fig, ax = plt.subplots(nrows=14, ncols=2,figsize=(25,45))



ax[9,0].barh(con_US2_wis['Admin Area'],con_US2_wis['Confirmed'],color='tab:orange' )

ax[9,0].set_title("Wisconsin")

ax[9,1].barh(con_US2_utah['Admin Area'],con_US2_utah['Confirmed'],color='tab:cyan' )

ax[9,1].set_title("Utah")



ax[10,0].barh(con_US2_vir['Admin Area'],con_US2_vir['Confirmed'],color=my_colors )

ax[10,0].set_title("Virginia")

ax[10,1].barh(con_US2_col['Admin Area'],con_US2_col['Confirmed'],color=my_colors )

ax[10,1].set_title("Colorado")



ax[11,0].barh(con_US2_ohio['Admin Area'],con_US2_ohio['Confirmed'],color=my_colors )

ax[11,0].set_title("Ohio")

ax[11,1].barh(con_US2_sda['Admin Area'],con_US2_sda['Confirmed'],color='tab:orange' )

ax[11,1].set_title("South Dakota")



ax[12,0].barh(con_US2_del['Admin Area'],con_US2_del['Confirmed'],color=my_colors )

ax[12,0].set_title("Delaware")

ax[12,1].barh(con_US2_rho['Admin Area'],con_US2_rho['Confirmed'],color=my_colors )

ax[12,1].set_title("Rhode Island")



ax[13,0].barh(con_US2_nca['Admin Area'],con_US2_nca['Confirmed'],color='tab:cyan' )

ax[13,0].set_title("North Carolina")

ax[13,1].barh(con_US2_ken['Admin Area'],con_US2_ken['Confirmed'],color='tab:pink' )

ax[13,1].set_title("Kentucky")





#before 18/4

ax[0,0].barh(con_US2_Cal['Admin Area'],con_US2_Cal['Confirmed'], color=my_colors)

ax[0,0].set_title("California")



#ax[0,1].barh(con_US2_NY['Admin Area'],con_US2_NY['Confirmed'],color=my_colors)

#ax[0,1].set_title("New York")



ax[0,1].barh(con_US2_mas['Admin Area'],con_US2_mas['Confirmed'],color=my_colors )

ax[0,1].set_title("Massachusetts")



ax[3,1].barh(con_US2_Mich['Admin Area'],con_US2_Mich['Confirmed'],color=my_colors )

ax[3,1].set_title("Michigan")

ax[2,0].barh(con_US2_ill['Admin Area'],con_US2_ill['Confirmed'],color=my_colors )

ax[2,0].set_title("Illinois")

ax[1,1].barh(con_US2_NJ['Admin Area'],con_US2_NJ['Confirmed'],color=my_colors )

ax[1,1].set_title("New Jersey")

ax[2,1].barh(con_US2_flo['Admin Area'],con_US2_flo['Confirmed'],color=my_colors )

ax[2,1].set_title("Florida")

ax[3,0].barh(con_US2_lou['Admin Area'],con_US2_lou['Confirmed'],color=my_colors )

ax[3,0].set_title("Louisiana")

ax[1,0].barh(con_US2_pen['Admin Area'],con_US2_pen['Confirmed'],color=my_colors )

ax[1,0].set_title("Pennsylvania")



ax[4,0].barh(con_US2_con['Admin Area'],con_US2_con['Confirmed'],color=my_colors )

ax[4,0].set_title("Connecticut")

ax[4,1].barh(con_US2_was['Admin Area'],con_US2_was['Confirmed'],color=my_colors )

ax[4,1].set_title("Washington State")



#ax[5,0].barh(con_US2_mas['Admin Area'],con_US2_mas['Confirmed'],color=my_colors )

#ax[5,0].set_title("Massachusetts")

ax[8,0].barh(con_US2_mis['Admin Area'],con_US2_mis['Confirmed'],color=my_colors )

ax[8,0].set_title("Missouri")

ax[5,1].barh(con_US2_tex['Admin Area'],con_US2_tex['Confirmed'],color=my_colors )

ax[5,1].set_title("Texas")



ax[6,0].barh(con_US2_ind['Admin Area'],con_US2_ind['Confirmed'],color=my_colors )

ax[6,0].set_title("Indiana")

ax[6,1].barh(con_US2_nev['Admin Area'],con_US2_nev['Confirmed'],color='tab:pink' )

ax[6,1].set_title("Nevada")



ax[7,0].barh(con_US2_ari['Admin Area'],con_US2_ari['Confirmed'],color=my_colors )

ax[7,0].set_title("Arizona")

ax[7,1].barh(con_US2_wasDC['Admin Area'],con_US2_wasDC['Confirmed'],color='tab:blue' )

ax[7,1].set_title("Washington D.C.")



ax[5,0].barh(con_US2_mar['Admin Area'],con_US2_mar['Confirmed'],color=my_colors )

ax[5,0].set_title("Maryland")

ax[8,1].barh(con_US2_geo['Admin Area'],con_US2_geo['Confirmed'],color=my_colors )

ax[8,1].set_title("Georgia")



#ax[9,0].barh(con_US2_wis['Admin Area'],con_US2_wis['Confirmed'],color='tab:green' )

#ax[9,0].set_title("Wisconsin")

#ax[9,1].barh(con_US2_geo['Admin Area'],con_US2_geo['Confirmed'],color=my_colors )

#ax[9,1].set_title("Georgia")



plt.ylabel('Cases')



fig.suptitle('US Areas with reported cases > 1000 (as of '+date_US+')',fontdict=font)

fig.tight_layout(pad=6.0)





#axs[0].plt.bar(con_US2['Admin Area'],con_US2['Confirmed'] )

#fig.autofmt_xdate(rotation=75)

plt.legend()

print("Other US States")
#calulates days until get > 50% population have COVID-19 for counties with >500 cases as at 26 March 2020

fifty = pd.DataFrame(columns=['Province/State','Country/Region','No. of Cases by','Date'])

table = tab.loc[tab["Confirmed"]>10]

#table = tab

fifty_rows = pd.DataFrame(columns=['SNo','Province/State','Country/Region','% Population','Days from 26Mar2020'])

count=0

count2=0

for a in range(len(table)):

   

    state = table.iloc[a,1]

    country = table.iloc[a,2]

    y=table.iloc[a,6]

    original = y

    d=table.iloc[a,14]

    p=table.iloc[a,5]

    count+=1

    for x in range(30):

        x+=1

        count2+=1

        cases = 2 * y        

        days_no = d * x

        pop = p *1000000

        y_perc = cases/pop*100

        fifty_rows.loc[count2] = [count2,state,country ,y_perc,days_no]

        if y_perc > 50:

            break

        y= cases

    start_date = datetime.date(2020, 3, 26)

    #no_days = date.timedelta(days_no)

    no_days = int(days_no)

    new_date = start_date + datetime.timedelta(no_days)  

    fifty.loc[count] = [state,country,int(cases),new_date]
from tabulate import tabulate



print("PREDICTED DATES FOR WHEN CONFIRMED CASES WILL PASS 50% OF THE POPULATION")

headers = ['Province/State','Country/Region','No. of Cases by','Date']

print(tabulate(fifty, headers=headers, showindex="never"))
aus = fifty_rows.loc[fifty_rows["Country/Region"]=="Australia"]

aus_all = aus.loc[aus["Province/State"]=="*"]

china = fifty_rows.loc[fifty_rows["Country/Region"]=="China"]

china_all = china.loc[china["Province/State"]=="*"]

italy = fifty_rows.loc[fifty_rows["Country/Region"]=="Italy"]

spain = fifty_rows.loc[fifty_rows["Country/Region"]=="Spain"]

#france = fifty_rows.loc[fifty_rows["Country/Region"]=="France"]

Korea_South = fifty_rows.loc[fifty_rows["Country/Region"]=="Korea, South"]

US_only= fifty_rows.loc[fifty_rows["Country/Region"]=="US"]

US_all = US_only.loc[US_only["Province/State"]=="*"]

singapore = fifty_rows.loc[fifty_rows["Country/Region"]=="Singapore"]

fig = plt.figure(figsize=(25,7))

plt.xlabel('Days after 26 March 2020')

plt.ylabel('Cases (% Population)')

plt.title("Predicted Exponential Growth in Confirmed Cases after 26 March 2020",fontdict=font)

plt.plot(aus_all['Days from 26Mar2020'],aus_all['% Population'],'tab:green',label="Australia")

plt.plot(china_all['Days from 26Mar2020'],china_all['% Population'],'tab:blue',label="China")



plt.plot(italy['Days from 26Mar2020'],italy['% Population'],'tab:red',label="Italy")

#plt.plot(france['Days from 26Mar2020'],france['% Population'],'tab:pink',label="France")



plt.plot(spain['Days from 26Mar2020'],spain['% Population'],'tab:purple',label="Spain")

plt.plot(US_all['Days from 26Mar2020'],US_all['% Population'],'tab:orange',label="US")

plt.plot(singapore['Days from 26Mar2020'],singapore['% Population'],'tab:cyan',label="Singapore")

plt.plot(Korea_South['Days from 26Mar2020'],Korea_South['% Population'],'tab:pink',label="South Korea")

plt.legend()

#plt.xlim((0,400))

#plt.ylim((0,100))



#fig.autofmt_xdate(rotation=75)

print("Italy, Spain, US, Singapore, South Korea, Australia and China")
country_df = pd.read_csv("../input/country-data/country_pop_data1.csv")

italy = country_df[country_df["Country/Region"]=="Italy"]

spain = country_df[country_df["Country/Region"]=="Spain"]

hubei = country_df[country_df["Province/State"]=="Hubei"]

china = country_df[country_df["Country/Region"]=="China"]

not_hubei= china[china["Province/State"]!="Hubei"]

japan = country_df[country_df["Country/Region"]=="Japan"]

s_korea = country_df[country_df["Country/Region"]=="Korea, South"]

sing = country_df[country_df["Country/Region"]=="Singapore"]

iran = country_df[country_df["Country/Region"]=="Iran"]

us = country_df[country_df["Country/Region"]=="US"]

uk = country_df[country_df["Country/Region"]=="United Kingdom"]

ger = country_df[country_df["Country/Region"]=="Germany"]
name = hubei

fig = plt.figure(figsize=(20,7))

plt.ylabel('Case Numbers')

plt.xlabel('Date')

plt.scatter(name["date"],name["Confirmed"],s=100,marker =8, c='tab:orange',label = 'Confirmed cases')

plt.scatter(name["date"],name["Active cases"],s=100,marker =8, c='tab:blue',label = 'Active cases')

plt.scatter(name["date"],name["New confirmed cases"],s=100,marker =8, c='tab:red',label = 'New cases(daily)')

plt.title("Case numbers for Hubei",fontdict=font)

fig.autofmt_xdate(rotation=75)

plt.legend()

print("Hubei, China")
name = s_korea

fig = plt.figure(figsize=(20,7))

plt.ylabel('Case Numbers')

plt.xlabel('Date')

plt.scatter(name["date"],name["Confirmed"],s=100,marker =8, c='tab:orange',label = 'Confirmed cases')

plt.scatter(name["date"],name["Active cases"],s=100,marker =8, c='tab:blue',label = 'Active cases')

plt.scatter(name["date"],name["New confirmed cases"],s=100,marker =8, c='tab:red',label = 'New cases(daily)')

plt.title("Case numbers for South Korea",fontdict=font)

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.xlim((20,120))

print(" South Korea")
name = italy

fig = plt.figure(figsize=(20,7))

plt.ylabel('Case Numbers')

plt.xlabel('Date')

plt.scatter(name["date"],name["Confirmed"],s=100,marker =8, c='tab:orange',label = 'Confirmed cases')

plt.scatter(name["date"],name["Active cases"],s=100,marker =8, c='tab:blue',label = 'Active cases')

plt.scatter(name["date"],name["New confirmed cases"],s=100,marker =8, c='tab:red',label = 'New cases (daily)')

plt.title("Case numbers for Italy",fontdict=font)

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.xlim((30,120))

print("Italy")
fig = plt.figure(figsize=(20,7))

plt.ylabel('New Cases Reported Daily')

plt.xlabel('Date')

plt.scatter(hubei["date"],hubei["New confirmed cases"],s=100,marker =8, c='tab:orange',label = 'Hubei')

plt.scatter(s_korea["date"],s_korea["New confirmed cases"],s=100,marker =8, c='tab:blue',label = 'South Korea')

plt.title("Changes in New Cases reported daily for South Korea and Hubei",fontdict=font)

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.ylim((-500,7000))

print("South Korea and Hubei China")


fig = plt.figure(figsize=(20,7))

plt.ylabel('New Cases Reported Daily')

plt.xlabel('Date')

plt.scatter(italy["date"],italy["New confirmed cases"],s=100,marker =8, c='tab:red',label = 'Italy')

plt.scatter(us["date"],us["New confirmed cases"],s=100,marker =8, c='tab:green',label = 'US')

plt.scatter(ger["date"],ger["New confirmed cases"],s=100,marker =8, c='tab:orange',label = 'Germany')

plt.scatter(spain["date"],spain["New confirmed cases"],s=100,marker =8, c='tab:blue',label = 'Spain')

plt.title("Changes in New Cases reported daily for Italy, Germany, Spain and the US",fontdict=font)

fig.autofmt_xdate(rotation=75)

plt.legend()

plt.xlim((30,120))

print("Italy, Germany, US, Singapore, Spain")