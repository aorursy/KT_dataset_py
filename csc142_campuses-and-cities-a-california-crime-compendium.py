# must for data analysis

% matplotlib inline

import numpy as np

import pandas as pd

from matplotlib.pyplot import *

import matplotlib.pyplot as plt

import seaborn as sns



# useful for data wrangling

import io, os, re, subprocess



# for sanity

from pprint import pprint
def ca_law_enforcement_by_campus(data_directory):

    filename = 'ca_law_enforcement_by_campus.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    header = ' '.join(lines[:6])

    header = re.sub('\n','',header)

    data = lines[6:]

    

    # Process each string in the list

    newlines = []

    for p in data:

        if( len(re.findall(',,,,',p))==0):

            newlines.append(p)



    # Combine into one long string, and do more processing

    one_string = '\n'.join(newlines)

    sio = io.StringIO(one_string)



    columnstr = header



    # Get rid of \r stuff

    columnstr = re.sub('\r',' ',columnstr)

    columnstr = re.sub('\s+',' ',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]



    df = pd.read_csv(sio,quotechar='"', names=columns, thousands=',')



    return df



def ca_offenses_by_campus(data_directory):

    filename = 'ca_offenses_by_campus.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    # Process each string in the list

    newlines = []

    for p in lines[1:]:

        if( len(re.findall(',,,,',p))==0):

            # This is a weird/senseless/badly formatted line

            if( len(re.findall('Medical Center, Sacramento5',p))==0):

                newlines.append(p)



    one_line = '\n'.join(newlines)

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio, quotechar='"', thousands=',', names=columns)

    

    return df



def ca_law_enforcement_by_city(data_directory):

    filename = 'ca_law_enforcement_by_city.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r',' ',content)

    [header,data] = content.split("civilians\"")

    header += "civilians\""



    data = data.strip()



    # Combine into one long string, and do more processing

    one_string = re.sub(r'([0-9]) ([A-Za-z])',r'\1\n\2',data)

    sio = io.StringIO(one_string)



    # Process column names

    columnstr = header.strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df





def ca_offenses_by_city(data_directory):

    filename = 'ca_offenses_by_city.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r','\n',content)



    lines = content.split('\n')

    one_line = '\n'.join(lines[1:])

    sio = io.StringIO(one_line)



    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df
def import_campus_df(data_directory):

    df_enforcement = ca_law_enforcement_by_campus(data_directory)

    df_offenses = ca_offenses_by_campus(data_directory)

    # Fix errant digits

    for r in df_offenses['Campus']:

        if(type(r)==type(' ')):

            df_offenses['Campus'][df_offenses['Campus']==r].map(lambda x : re.sub(r'[0-9]$','',x))

    df_campus = pd.merge(df_offenses, df_enforcement, 

                     on=[df_enforcement.columns[0],df_enforcement.columns[1],df_enforcement.columns[2]])

    

    # Useful derived quantities

    df_campus['Per Capita Law Enforcement Personnel'] = (df_campus['Total law enforcement employees'])/(df_campus['Student enrollment'])

    df_campus['Law Enforcement Civilians Per Officer'] = (df_campus['Total civilians'])/(df_campus['Total officers'])

    df_campus['Aggregate Crime'] = df_campus['Violent crime'] + df_campus['Property crime'] + df_campus['Arson']

    df_campus['Per Capita Violent Crime'] = (df_campus['Violent crime'])/(df_campus['Student enrollment'])

    df_campus['Per Capita Property Crime'] = (df_campus['Property crime'])/(df_campus['Student enrollment'])

    df_campus['Per Capita Aggregate Crime'] = (df_campus['Violent crime'] + df_campus['Property crime'] + df_campus['Arson'])/(df_campus['Student enrollment'])

    df_campus['Aggregate Crime Per Officer'] = (df_campus['Aggregate Crime'])/(df_campus['Total officers'])

    df_campus['Violent Crime Per Officer'] = (df_campus['Violent crime'])/(df_campus['Total officers'])

    df_campus['Property Crime Per Officer'] = (df_campus['Property crime'])/(df_campus['Total officers'])

    

    return df_campus
df_campus = import_campus_df('../input')
df_city_enforcement = ca_law_enforcement_by_city('../input')

df_city_offenses = ca_offenses_by_city('../input')
df_city_enforcement.head()
df_city_offenses.head()
print(df_city_offenses['City'].shape)
df_city = pd.merge(df_city_offenses, df_city_enforcement, 

                 on=[df_city_enforcement.columns[0],df_city_enforcement.columns[1]])
df_city.head()
print(df_city['City'][df_city['Violent crime']==max(df_city['Violent crime'])])



print(df_city['City'][df_city['Property crime']==max(df_city['Property crime'])])
# Stuff it all in a function:



def import_city_df(data_directory):

    df_city_enforcement = ca_law_enforcement_by_city(data_directory)

    df_city_offenses = ca_offenses_by_city(data_directory)

    

    # Fix errant digits

    df_city_offenses['City'] = df_city_offenses['City'].map(lambda x : re.sub(r'[0-9]$','',x))

    

    df_city = pd.merge(df_city_offenses, df_city_enforcement, 

                       on=[df_city_enforcement.columns[0],df_city_enforcement.columns[1]])

    

    # Useful derived quantities

    df_city['Aggregate Crime'] = df_city['Violent crime'] + df_city['Property crime']

    df_city['Per Capita Violent Crime'] = df_city['Violent crime']/df_city['Population']

    df_city['Per Capita Property Crime'] = df_city['Property crime']/df_city['Population']

    df_city['Per Capita Aggregate Crime'] = df_city['Aggregate Crime']/df_city['Population']



    df_city['Per Capita Law Enforcement Personnel'] = (df_city['Total law enforcement employees'])/(df_city['Population'])



    df_city['Aggregate Crime Per Officer'] = (df_city['Aggregate Crime'])/(df_city['Total officers'])

    df_city['Violent Crime Per Officer'] = (df_city['Violent crime'])/(df_city['Total officers'])

    df_city['Property Crime Per Officer'] = (df_city['Property crime'])/(df_city['Total officers'])



    return df_city
fig, ax2 = subplots(1, figsize=(8,4))



sns.distplot(df_city['Population'][df_city['Violent crime']>1].map(lambda x : np.log10(x)), label='Population', ax=ax2)

sns.distplot(df_city['Violent crime'][df_city['Violent crime']>0].map( lambda x : np.log10(x) ), label='Violent', ax=ax2)

sns.distplot(df_city['Property crime'][df_city['Property crime']>0].map( lambda x : np.log10(x) ), label='Property', ax=ax2)



ax2.set_xlabel('Log Number (Pwr Of 10)')

ax2.set_title('Log Population/Crime Rate')

ax2.legend()



show()
import scipy.stats as stats

import statsmodels.api as sm
fig1, ax1 = subplots(1,1,figsize=(6,4))

fig2, ax2 = subplots(1,1,figsize=(6,4))



vcqq = sm.qqplot(df_city['Violent crime'][df_city['Violent crime']>0].map( lambda x : np.log10(x) ), 

            fit=True, line='45', ax=ax1)

ax1.set_title('Quantile-Quantile Plot: Violent Crime')



pcqq = sm.qqplot(df_city['Property crime'][df_city['Property crime']>0].map( lambda x : np.log10(x) ), 

            fit=True, line='45', ax=ax2)

ax2.set_title('Quantile-Quantile Plot: Property Crime')



plt.show()
vcqq = stats.probplot(df_city['Violent crime'][df_city['Violent crime']>0].map( lambda x : np.log10(x) ), dist="norm")



# len: 327

vcqqx = vcqq[0][0]

vcqqy = vcqq[0][1]



cut1 = 15

cut2 = 205

plt.plot(vcqqx[:cut1],vcqqy[:cut1],'o', color=sns.xkcd_rgb["dusty green"])

plt.plot(vcqqx[cut1:cut2],vcqqy[cut1:cut2],'o', color=sns.xkcd_rgb["denim blue"])

plt.plot(vcqqx[cut2:],vcqqy[cut2:],'o', color=sns.xkcd_rgb["watermelon"])

plt.title('Violent Crime: Quantile-Quantile Color Coded for Outliers')

plt.show()
print("%0.3e"%(df_city['Population'].sum()))
print("="*40)

print("Unusually high total incidence of violent crime:")

print(df_city[['City','Population','Violent crime']].sort_values('Violent crime',ascending=False)[:cut1])
print("="*40)

print("Unusually low total incidence of violent crime:")

print(df_city[['City','Population','Violent crime']].sort_values('Violent crime',ascending=True)[:df_city.shape[0]-cut2])
pcqq = stats.probplot(df_city['Property crime'][df_city['Property crime']>1].map( lambda x : np.log10(x) ), dist='norm')



pcqqx = pcqq[0][0]

pcqqy = pcqq[0][1]



cut1 = 6

cut2 = 204

plt.plot(pcqqx[:cut1],pcqqy[:cut1],'o', color=sns.xkcd_rgb["dusty green"])

plt.plot(pcqqx[cut1:cut2],pcqqy[cut1:cut2],'o', color=sns.xkcd_rgb["denim blue"])

plt.plot(pcqqx[cut2:],pcqqy[cut2:],'o', color=sns.xkcd_rgb["watermelon"])

plt.title('Property Crime: Quantile-Quantile Color Coded for Outliers')

plt.show()
print("="*40)

print("Unusually high total incidence of property crime:")

print(df_city[['City','Population','Property crime']].sort_values('Property crime',ascending=False)[:cut1])
# Useful derived quantities

df_city['Aggregate Crime'] = df_city['Violent crime'] + df_city['Property crime']

df_city['Per Capita Violent Crime'] = df_city['Violent crime']/df_city['Population']

df_city['Per Capita Property Crime'] = df_city['Property crime']/df_city['Population']

df_city['Per Capita Aggregate Crime'] = df_city['Aggregate Crime']/df_city['Population']



df_city['Per Capita Law Enforcement Personnel'] = (df_city['Total law enforcement employees'])/(df_city['Population'])



df_city['Aggregate Crime Per Officer'] = (df_city['Aggregate Crime'])/(df_city['Total officers'])

df_city['Violent Crime Per Officer'] = (df_city['Violent crime'])/(df_city['Total officers'])

df_city['Property Crime Per Officer'] = (df_city['Property crime'])/(df_city['Total officers'])
fig, ax1 = subplots(1, figsize=(8,4))



sns.distplot(df_city['Per Capita Violent Crime'][df_city['Violent crime']>0].map( lambda x : np.log10(x) ), label='Violent', ax=ax1)

sns.distplot(df_city['Per Capita Property Crime'][df_city['Property crime']>0].map( lambda x : np.log10(x) ), label='Property', ax=ax1)



ax1.set_xlabel('Log Number (Pwr Of 10)')

ax1.set_title('Log Population/Crime Rate')

ax1.legend()



show()
vcqq = stats.probplot(df_city['Per Capita Violent Crime'][df_city['Violent crime']>0].map( lambda x : np.log10(x) ), dist="norm")



# len: 327

vcqqx = vcqq[0][0]

vcqqy = vcqq[0][1]



cut1 = 3

cut2 = 214

plot(vcqqx[:cut1],vcqqy[:cut1],'o', color=sns.xkcd_rgb["dusty green"])

plot(vcqqx[cut1:cut2],vcqqy[cut1:cut2],'o', color=sns.xkcd_rgb["denim blue"])

plot(vcqqx[cut2:],vcqqy[cut2:],'o', color=sns.xkcd_rgb["watermelon"])

title('Per Capita Violent Crime: Quantile-Quantile Color Coded for Outliers')

show()
print("="*40)

print("Unusually high per capita incidence of property crime:")

print(df_city[['City','Population','Per Capita Property Crime']].sort_values('Per Capita Property Crime',ascending=False)[:cut1])

print("="*40)

print("Unusually low per capita incidence of property crime:")

print(df_city[['City','Population','Per Capita Property Crime']].sort_values('Per Capita Property Crime',ascending=True)[:df_city.shape[0]-cut2])
df_campus.head()
z = pd.DataFrame([])

for (i,row) in df_campus.iterrows():

    if row['Campus'] is not np.nan:

        city = row['Campus']

        if city in df_city['City'].values:

            z = pd.concat([ z, df_city[df_city['City']==city] ])



print(len(z))

print(z['City'])
campus_matches = pd.DataFrame([])

for (i,row) in df_campus.iterrows():

    if row['Campus'] is not np.nan:

        city = row['Campus']

        if city in df_city['City'].values:

            # Merge DataFrames:

            # part 1: row

            # part 2: df_city[df_city['City']==city]

            ff = row.to_frame().transpose()

            merged = pd.merge( ff, df_city[df_city['City']==city], left_on='Campus', right_on='City', suffixes=(' campus',' city'))

            campus_matches = pd.concat([ campus_matches, merged ])



print(len(campus_matches))

print(campus_matches.columns)

print(campus_matches[['University/College','Campus']])
g = sns.jointplot(x="Per Capita Aggregate Crime campus",y="Per Capita Aggregate Crime city",data=campus_matches)
filtered_campus_matches = campus_matches.sort_values('Per Capita Aggregate Crime campus',ascending=False)[1:]

g = sns.jointplot(x="Per Capita Aggregate Crime campus",y="Per Capita Aggregate Crime city",data=filtered_campus_matches)
fig = figure(figsize=(10,4))



ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



ax1.plot(filtered_campus_matches['Per Capita Aggregate Crime campus'], 

         filtered_campus_matches['Per Capita Aggregate Crime city'],

         'o')

ax1.set_xlabel('Per Capita Crime Rate, Campus')

ax1.set_ylabel('Per Capita Crime Rate City')

ax1.set_title('City vs Campus Crime Rates')



ax2.plot(campus_matches['Per Capita Aggregate Crime campus'],

         campus_matches['Per Capita Aggregate Crime city'],

         'o')

ax2.set_xlabel('Per Capita Crime Rate, Campus')

ax2.set_ylabel('Per Capita Crime Rate City')

ax2.set_title('City vs Campus Crime Rates, Extended Axis')



show()
fields = ['University/College','Campus','Per Capita Aggregate Crime campus']

campus_matches[fields].sort_values(fields[2],ascending=False)[:3]
# city to campus crime ratio: 

# large means city is more dangerous than campus. 

# small is very bad.

campus_matches['City to Campus Per Capita Crime Ratio'] = campus_matches['Per Capita Aggregate Crime city']/campus_matches['Per Capita Aggregate Crime campus']

campus_matches['City to Campus Per Capita Violent Crime Ratio'] = campus_matches['Per Capita Violent Crime city']/campus_matches['Per Capita Violent Crime campus']

campus_matches['City to Campus Per Capita Property Crime Ratio'] = campus_matches['Per Capita Property Crime city']/campus_matches['Per Capita Property Crime campus']



# city to campus population ratio:

# larger means, big city and small campus. UCSF.

# small means, small city and big campus. Davis.

campus_matches['City to Campus Population Ratio'] = campus_matches['Population']/campus_matches['Student enrollment']

campus_matches['City to Campus Log Population Ratio'] = campus_matches['City to Campus Population Ratio'].map(lambda x : np.log10(x))
f, ax1 = subplots(1,1, figsize=(6,4))

f, ax2 = subplots(1,1, figsize=(6,4))

f, ax3 = subplots(1,1, figsize=(6,4))



# ----------

# plot 1

# Student enrollment vs per capita crime rate (city and campus)

ax1.semilogx(campus_matches['Student enrollment'],campus_matches['Per Capita Aggregate Crime city'],'o',label='City')

ax1.semilogx(campus_matches['Student enrollment'],campus_matches['Per Capita Aggregate Crime campus'],'o',label='Campus')



ax1.legend()

ax1.set_xlabel('Student Population (k)')

ax1.set_ylabel('Per Capita Crime Rate')

ax1.set_title('Population vs. Aggregate Crime Rate')



# -----------

# plot 2

# City:campus population ratio vs City:campus per capita crime ratio

# clipped x axis

ax2.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'])

#ax2.semilogy(campus_matches['City to Campus Population Ratio'],

#             campus_matches['City to Campus Per Capita Violent Crime Ratio'],

#             'o', color=sns.xkcd_rgb['muted blue'])



ax2.set_xlabel('City:Campus Ratio: Population')

ax2.set_ylabel('City:Campus Ratio: Per Capita Aggregate Crime')

ax2.set_xlim([0,50])

ax2.set_title('Population Ratio vs. Aggregate Crime Rate Ratio')



# ------------

# plot 3

# City:campus population ratio vs City:campus per capita crime ratio

ax3.semilogy(campus_matches['City to Campus Population Ratio'][campus_matches['University/College'].ne('University of California')],

             campus_matches['City to Campus Per Capita Crime Ratio'][campus_matches['University/College'].ne('University of California')],

             'o', color=sns.xkcd_rgb['faded red'], label='Non-UC')

ax3.semilogy(campus_matches['City to Campus Population Ratio'][campus_matches['University/College']=='University of California'],

             campus_matches['City to Campus Per Capita Crime Ratio'][campus_matches['University/College']=='University of California'],

             'o', color=sns.xkcd_rgb['faded purple'], label='Univ of Calif')



ax3.legend(loc='best')

ax3.set_xlabel('City:Campus Ratio: Population')

ax3.set_ylabel('City:Campus Ratio: Per Capita Aggregate Crime')

ax3.set_title('Population Ratio vs. Crime Rate Ratio, Extended')



show()
fields = ['University/College','Campus','Per Capita Aggregate Crime campus']

campus_matches[fields].sort_values(fields[2],ascending=False)[:3]
# city to campus crime ratio: 

# large means city is more dangerous than campus. 

# small is very bad.

campus_matches['City to Campus Per Capita Crime Ratio'] = campus_matches['Per Capita Aggregate Crime city']/campus_matches['Per Capita Aggregate Crime campus']

campus_matches['City to Campus Per Capita Violent Crime Ratio'] = campus_matches['Per Capita Violent Crime city']/campus_matches['Per Capita Violent Crime campus']

campus_matches['City to Campus Per Capita Property Crime Ratio'] = campus_matches['Per Capita Property Crime city']/campus_matches['Per Capita Property Crime campus']



# city to campus population ratio:

# larger means, big city and small campus. UCSF.

# small means, small city and big campus. Davis.

campus_matches['City to Campus Population Ratio'] = campus_matches['Population']/campus_matches['Student enrollment']

campus_matches['City to Campus Log Population Ratio'] = campus_matches['City to Campus Population Ratio'].map(lambda x : np.log10(x))
f1, ax1 = subplots(1,1,figsize=(6,4))

f2, ax2 = subplots(1,1,figsize=(6,4))

f3, ax3 = subplots(1,1,figsize=(6,4))



# ----------

# plot 1

# Student enrollment vs per capita crime rate (city and campus)

ax1.semilogx(campus_matches['Student enrollment'],campus_matches['Per Capita Aggregate Crime city'],'o',label='City')

ax1.semilogx(campus_matches['Student enrollment'],campus_matches['Per Capita Aggregate Crime campus'],'o',label='Campus')



ax1.legend()

ax1.set_xlabel('Student Population (k)')

ax1.set_ylabel('Per Capita Crime Rate')

ax1.set_title('Population vs. Aggregate Crime Rate')



# -----------

# plot 2

# City:campus population ratio vs City:campus per capita crime ratio

# clipped x axis

ax2.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'])

#ax2.semilogy(campus_matches['City to Campus Population Ratio'],

#             campus_matches['City to Campus Per Capita Violent Crime Ratio'],

#             'o', color=sns.xkcd_rgb['muted blue'])



ax2.set_xlabel('City:Campus Ratio: Population')

ax2.set_ylabel('City:Campus Ratio: Per Capita Aggregate Crime')

ax2.set_xlim([0,50])

ax2.set_title('Population Ratio vs. Aggregate Crime Rate Ratio')



# ------------

# plot 3

# City:campus population ratio vs City:campus per capita crime ratio

ax3.semilogy(campus_matches['City to Campus Population Ratio'][campus_matches['University/College'].ne('University of California')],

             campus_matches['City to Campus Per Capita Crime Ratio'][campus_matches['University/College'].ne('University of California')],

             'o', color=sns.xkcd_rgb['faded red'], label='Non-UC')

ax3.semilogy(campus_matches['City to Campus Population Ratio'][campus_matches['University/College']=='University of California'],

             campus_matches['City to Campus Per Capita Crime Ratio'][campus_matches['University/College']=='University of California'],

             'o', color=sns.xkcd_rgb['faded purple'], label='Univ of Calif')



ax3.legend(loc='best')

ax3.set_xlabel('City:Campus Ratio: Population')

ax3.set_ylabel('City:Campus Ratio: Per Capita Aggregate Crime')

ax3.set_title('Population Ratio vs. Crime Rate Ratio, Extended')



show()
fields = ['University/College','Campus','City to Campus Population Ratio']

campus_matches[fields].sort_values(fields[2],ascending=True)[:4]
fields = ['University/College','Campus','City to Campus Per Capita Crime Ratio']

campus_matches[fields].sort_values(fields[2],ascending=True)[:4]
f, (ax1,ax2) = subplots(1,2, figsize=(10,4))



# -----------

# plot 2

# City:campus population ratio vs City:campus per capita crime ratio

# clipped x axis

ax1.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted purple'],

            label='Aggregate')

ax1.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Violent Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'],

             label='Violent')

ax1.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Property Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted blue'],

             label='Property')

ax1.set_xlim([0,50])





ax2.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted purple'],

            label='Aggregate')

ax2.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Violent Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'],

             label='Violent')

ax2.semilogy(campus_matches['City to Campus Population Ratio'],

             campus_matches['City to Campus Per Capita Property Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted blue'],

             label='Property')



ax1.legend(loc='best')

ax1.set_xlabel('City:Campus Ratio: Population')

ax1.set_ylabel('City:Campus Ratio: Per Capita Crime')

ax1.set_title('Population Ratio vs. Crime Rate Ratio')



ax2.legend(loc='best')

ax2.set_xlabel('City:Campus Ratio: Population')

ax2.set_ylabel('City:Campus Ratio: Per Capita Crime')

ax2.set_title('(Same Plot, Extended Axis)')



show()
fields = ['University/College','Campus','City to Campus Per Capita Violent Crime Ratio']

campus_matches[fields].sort_values(fields[2],ascending=True)[:4]
fields = ['University/College','Campus','City to Campus Per Capita Property Crime Ratio']

campus_matches[fields].sort_values(fields[2],ascending=True)[:4]
df_campus[['University/College','Campus']].head()
list_o_cities = ["Santa Barbara",

                 "Pomona",

                 "Bakersfield",

                 "Santa Barbara",

                 "Los Angeles",

                 "Hayward",

                 "Fresno",

                 "Santa Cruz",

                 "Los Angeles",

                 "Sacramento",

                 "San Bernardino",

                 "San Diego",

                 "Turlock",

                 "Fresno",

                 "San Pablo",

                 "Los Angeles",

                 "Sunnyvale",

                 "Arcata",

                 "San Rafael",

                 "Riverside",

                 "San Bernardino",

                 "San Diego",

                 "San Francisco",

                 "Cotati",

                 "Fresno",

                 "Berkeley",

                 "Davis",

                 "San Francisco",

                 "Los Angeles",

                 "Merced",

                 "Riverside",

                 "San Diego",

                 "San Francisco",

                 "Santa Barbara",

                 "Santa Cruz",

                 "Ventura"

         ]

s = pd.Series(list_o_cities)

df_campus['City'] = s
# Check if any cities are not found in the list of cities we have crime stats on

# (should output nothing)

for i in df_campus['City'].values:

    if i not in df_city['City'].values:

        print(i)
new_campus_matches = pd.DataFrame([])

for (i,row) in df_campus.iterrows():

    # Merge DataFrames:

    # part 1: row

    # part 2: df_city[df_city['City']==city]

    ff = row.to_frame().transpose()

    city = row['City']

    merged = pd.merge( ff, df_city[df_city['City']==city], on='City',

                      suffixes=(' campus',' city'))

    new_campus_matches = pd.concat([ new_campus_matches, merged ])
# Now we have a big list of joint campus/city statistics:

print(len(new_campus_matches))

new_campus_matches.head()
g = sns.jointplot(x="Per Capita Aggregate Crime campus",y="Per Capita Aggregate Crime city",data=new_campus_matches)
filtered_new_campus_matches = new_campus_matches[new_campus_matches['Per Capita Aggregate Crime campus']<0.025]

g = sns.jointplot(x="Per Capita Aggregate Crime campus",y="Per Capita Aggregate Crime city",data=filtered_new_campus_matches)
fig = figure(figsize=(10,4))



ax1 = fig.add_subplot(121)

ax1.plot(new_campus_matches['Per Capita Aggregate Crime campus'], 

         new_campus_matches['Per Capita Aggregate Crime city'],

         'o')

ax1.set_xlabel('Per Capita Crime Rate, Campus')

ax1.set_ylabel('Per Capita Crime Rate City')

ax1.set_xlim([0.0,0.02])



ax2 = fig.add_subplot(122)

ax2.plot(new_campus_matches['Per Capita Aggregate Crime campus'], 

         new_campus_matches['Per Capita Aggregate Crime city'],

         'o')

ax2.set_xlabel('Per Capita Crime Rate, Campus')

ax2.set_ylabel('Per Capita Crime Rate City')





show()
fields = ['University/College','Campus','Per Capita Aggregate Crime campus']

new_campus_matches[fields].sort_values(fields[2],ascending=False)[:4]
# city to campus crime ratio:

# large means city is more dangerous than campus.

# small is very bad.

new_campus_matches['City to Campus Per Capita Crime Ratio'] = new_campus_matches['Per Capita Aggregate Crime city']/(new_campus_matches['Per Capita Aggregate Crime campus']+0.0001)

new_campus_matches['City to Campus Per Capita Violent Crime Ratio'] = new_campus_matches['Per Capita Violent Crime city']/(new_campus_matches['Per Capita Violent Crime campus']+0.0001)

new_campus_matches['City to Campus Per Capita Property Crime Ratio'] = new_campus_matches['Per Capita Property Crime city']/(new_campus_matches['Per Capita Property Crime campus']+0.0001)



# city to campus population ratio:

# larger means, big city and small campus. UCSF.

# small means, small city and big campus. Davis.

new_campus_matches['City to Campus Population Ratio'] = new_campus_matches['Population']/new_campus_matches['Student enrollment']

new_campus_matches['City to Campus Log Population Ratio'] = new_campus_matches['City to Campus Population Ratio'].map(lambda x : np.log10(x))
# city to campus crime ratio:

# large means city is more dangerous than campus.

# small is very bad.

new_campus_matches['Campus to City Per Capita Crime Ratio'] = new_campus_matches['Per Capita Aggregate Crime campus']/new_campus_matches['Per Capita Aggregate Crime city']

new_campus_matches['Campus to City Per Capita Violent Crime Ratio'] = new_campus_matches['Per Capita Violent Crime campus']/new_campus_matches['Per Capita Violent Crime city']

new_campus_matches['Campus to City Per Capita Property Crime Ratio'] = new_campus_matches['Per Capita Property Crime campus']/new_campus_matches['Per Capita Property Crime city']



# city to campus population ratio:

# larger means, big city and small campus. UCSF.

# small means, small city and big campus. Davis.

new_campus_matches['Campus to City Population Ratio'] = new_campus_matches['Student enrollment']/new_campus_matches['Population']

new_campus_matches['Campus to City Log Population Ratio'] = new_campus_matches['Campus to City Population Ratio'].map(lambda x : np.log10(x))
f, (ax1,ax2) = subplots(1,2, figsize=(10,4))



# -----------

# plot 1

ax1.semilogy(new_campus_matches['Campus to City Log Population Ratio'],

             new_campus_matches['Campus to City Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted purple'],

            label='Aggregate')

ax1.semilogy(new_campus_matches['Campus to City Log Population Ratio'],

             new_campus_matches['Campus to City Per Capita Violent Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'],

             label='Violent')

ax1.semilogy(new_campus_matches['Campus to City Log Population Ratio'],

             new_campus_matches['Campus to City Per Capita Property Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted blue'],

             label='Property')



ax1.set_xlabel('Campus:City Log Ratio: Population')

ax1.set_ylabel('Campus:City Ratio: Per Capita Crime')

ax1.set_title('Population Ratio vs. Crime Rate Ratio')

ax1.legend(loc='best')



# -----------

# plot 2

# City:campus population ratio vs City:campus per capita crime ratio

# clipped x axis

ax2.semilogy(new_campus_matches['Campus to City Log Population Ratio'],

             new_campus_matches['Campus to City Per Capita Crime Ratio'],

             'o', color=sns.xkcd_rgb['muted pink'])



ax2.set_xlabel('Campus:City Ratio: Population')

ax2.set_ylabel('Campus:City Ratio: Per Capita Aggregate Crime')

ax2.set_title('Population Ratio vs. Aggregate Crime Rate Ratio')



show()
name_fields = ['University/College','Campus','City']

fields = ['Campus to City Log Population Ratio','Campus to City Per Capita Crime Ratio']



low = new_campus_matches[name_fields+fields][new_campus_matches[fields[0]]<-1.0]

low['Campus-City Ratio']='Campus < 10% City'

low = low.sort_values(fields[0],ascending=True)



hi = new_campus_matches[name_fields+fields][new_campus_matches[fields[0]]>-1.0]

hi['Campus-City Ratio']='Campus > 10% City'

hi = hi.sort_values(fields[0],ascending=True)



both = pd.concat([low,hi])

both = both.sort_values(fields[0],ascending=True)
sns.lmplot(data=both, x=fields[0], y=fields[1],hue='Campus-City Ratio')

show()
import statsmodels.api as sm
# ----------

# Build the linear model for 

# low campus pop/city pop ratios



# floats and numpy 64 floats don't play well together...???

lowx_names = low[name_fields+fields]

lowx = low[fields[0]].values

lowy = low[fields[1]].map(lambda b : np.float64(b)).values



# super handy method

lowx = sm.add_constant(lowx)



lowlm = sm.OLS(lowy,lowx).fit()



# ----------

# Build the linear model for 

# hi campus pop/city pop ratios



hix_names = hi[name_fields+fields]

hix = hi[fields[0]].values

hiy = hi[fields[1]].map(lambda b : np.float64(b)).values



# handy

hix = sm.add_constant(hix)



hilm = sm.OLS(hiy, hix).fit()
lowlm.summary()
hilm.summary()
# Create a grid of x values for regression line (xprime, yprime)

lowxprime = np.linspace( -3, 1, 100)#np.min(lowx), np.max(lowx) )

lowxprime = sm.add_constant(lowxprime)

lowyprime = lowlm.predict(lowxprime)



# Calculate residuals for original x values

lowresid = lowy - lowlm.predict( sm.add_constant(lowx) )



# Create a grid of x values for regression line (xprime, yprime)

hixprime = np.linspace( -3, 1, 100)#np.min(hix), np.max(hix), 100 )

hixprime = sm.add_constant(hixprime)

hiyprime = hilm.predict(hixprime)



# Calculate residuals for original x values

hiresid = hiy - hilm.predict( sm.add_constant(hix) )
# Oh, and colors too

colorz = ['dusty blue','dusty green']

colors = [sns.xkcd_rgb[z] for z in colorz]
# Now visualize everything we've assembled:

fig = figure(figsize=(6,4))



ax1 = fig.add_subplot(111)

#ax2 = fig.add_subplot(122)



ax1.plot(lowx[:,1],lowy,'o', color=colors[0])

ax1.plot(lowxprime[:,1], lowyprime,'-', color=colors[0])



ax1.plot(hix[:,1],hiy,'o', color=colors[1])

ax1.plot(hixprime[:,1], hiyprime,'-', color=colors[1])



ax1.set_xlabel('Campus to City Log Population Ratio')

ax1.set_ylabel('Campus to City Per Capita Crime Rate Ratio')



show()
# More colors

quantile_colorz = ['dusty green','denim blue','watermelon']

quantile_colors = [sns.xkcd_rgb[z] for z in quantile_colorz]
# Get the low side linear model residual quantiles (UCSF/UC Hastings will be the huge outliers)

lowqq = stats.probplot( lowresid, dist='norm' )

lowqqx = lowqq[0][0]

lowqqy = lowqq[0][1]



# Get the hi side linear model residual quantiles

hiqq = stats.probplot( hiresid, dist='norm' )

hiqqx = hiqq[0][0]

hiqqy = hiqq[0][1]
# The cuts are where colors change (used to highlight outliers)

fig = figure(figsize=(6,4))

ax = fig.add_subplot(111)



cut1 = 0

cut2 = 20

ax.plot(lowqqx[:cut1],    lowqqy[:cut1],    'o', color=quantile_colors[0])

ax.plot(lowqqx[cut1:cut2],lowqqy[cut1:cut2],'o', color=quantile_colors[1])

ax.plot(lowqqx[cut2:],    lowqqy[cut2:],    'o', color=quantile_colors[2])



ax.set_xlabel('Normal Distribution')

ax.set_ylabel('Data')

ax.set_title('Quantile-Quantile of Residuals for Population Ratio-Crime Ratio\nLow School:City Pop. Ratio')

show()
lowx_names.sort_values('Campus to City Per Capita Crime Ratio',ascending=False)[:3]
# The cuts are where colors change (used to highlight outliers)

fig = figure(figsize=(6,4))

ax = fig.add_subplot(111)



cut1 = 6

cut2 = 11

ax.plot(hiqqx[:cut1],    hiqqy[:cut1],    'o', color=quantile_colors[0])

ax.plot(hiqqx[cut1:cut2],hiqqy[cut1:cut2],'o', color=quantile_colors[1])

ax.plot(hiqqx[cut2:],    hiqqy[cut2:],    'o', color=quantile_colors[2])



ax.set_xlabel('Normal Distribution')

ax.set_ylabel('Data')

ax.set_title('Quantile-Quantile of Residuals for Population Ratio-Crime Ratio\nHi School:City Pop. Ratio')

show()
sns.distplot(hiqqy,bins=5)

xlabel('Residual y - yhat')
print(hix_names.sort_values('Campus to City Per Capita Crime Ratio',ascending=False)[:3])
print(hix_names.sort_values('Campus to City Per Capita Crime Ratio',ascending=True)[:3])
hix_names['Resid'] = hiresid

lowx_names['Resid'] = lowresid
#print hix_names.columns

#print new_campus_matches.head()



#print hix_names.columns



z = pd.merge(hix_names,new_campus_matches,how='inner')

z['Log Population'] = z['Population'].map(lambda x : np.log10(x))
print(z.columns)
fields = ['Resid',

          'Per Capita Law Enforcement Personnel campus',

          'Student enrollment',

          'Population']



plt.figure(figsize=(6,6))

g = sns.PairGrid(z[fields])

#g.map(scatter)

g.map_diag(hist)

g.map_offdiag(scatter);

plt.show()
qqdat = z.sort_values('Aggregate Crime Per Officer campus')['Resid']



# Get the hi side linear model residual quantiles

hiqq = stats.probplot( hiresid, dist='norm' )

hiqqx = hiqq[0][0]

hiqqy = hiqq[0][1]

plt.show()