import matplotlib.pyplot as plt

import geopandas as gpd

import networkx as nx

import pandas as pd

import numpy as np

import matplotlib

import datetime, json, glob, unidecode

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.rcParams.update({'font.size': 17, 'lines.linewidth':4})
# Updated confirmed cases in mexico

c_pday = [3,4,5,5,5,5,5,6,6,7,7,7,11,15,26,41,53,82,93,118,164,203,251,316,367,405,475,585,717,848,993,1094,1215,1378,1510,1688,1890,2143,2439,2785,3181,3441,

          3844,4219,4661,5014,5399,5847,6297,6875,7497,8261,8772,9501,10544,11633,12872,13842,14677,15529,16752,17799,19224,20739,22088,

          23471,24905,26025,27634,29616,31522,33460,35022,36327,38324,40186,42595,45032,47144,49219,51633,54346,

          56594,59567,62527,65856,68620,71105,74560,78023,81400,84627,87512,90664,93435,97326,101238,105680,110026,

          113619,117103,120102,124301,129184,133974,139196,142690,146837,150264,154863,159793,165455,170485,175202,180545,

          185122,191410,196847,202951,208392,212802,216852,220657,226089,231770,238511,245251,252165,256848,261750,268008,275003,

          282283,289174,295268,299750,304435,311486,317635,324041,331298,338913,344224,349396,356255,362274,370712]

mx_confirmed_cases = np.array(c_pday)



def get_date_list(base, total=len(mx_confirmed_cases)):

    return [(base - datetime.timedelta(days=x)).strftime("%d-%b-%Y") for x in range(total)][::-1]



# Create data frame

mx_covid = pd.DataFrame(mx_confirmed_cases, columns=['Confirmed Cases'])



# Confirmed deads

d_pday = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,5,6,8,12,16,20,28,29,37,50,60,79,94,125,141,174,194,233,273,296,332,406,449,486,546,650,686,712,

          857,970,1069,1221,1305,1351,1434,1569,1732,1859,1972,2061,2154,2270,2507,2704,2961,3160,3353,

          3465,3573,3926,4220,4477,4767,5045,5177,5332,5666,6090,6510,6989,7179,7394,7633,

          8134,8597,9044,9415,9779,9930,10167,10637,11728,12545,13170,13511,13699,14053,14649,15357,

          15944,16448,16872,17141,17580,18310,19080,19747,20394,20781,21825,22584,23377,24324,25060,25779,26381,26648,27121,27769,

          28510,29189,29843,30366,30639,31119,32014,32796,33526,34191,34730,35006,35491,36327,36906,37574,38310,38888,39184,39485,

          40400,41190,41908]

mx_covid['Deceased'] = d_pday



# Get the dates for the confirmed cases

date_list = get_date_list(datetime.datetime.today() - datetime.timedelta(days=1))

mx_covid['Dates'] = date_list





# Save data frame

mx_covid.to_csv('covid_mx.csv',index=False)

# mx_covid.head()
# --------------

# Mexico .shp

# --------------

path = '/kaggle/input/mxstatesdataset/Mexico_States.shp'

data = gpd.read_file(path)

data['NAME'] = data['NAME'].str.lower()

# ---------------

# Confirmed cases

# ---------------

df = pd.read_csv('/kaggle/input/covid19-mx/casos_confirmados.csv')

df = df.dropna()

df.head()

center_states = ['distrito federal', 'querétaro', 'puebla', 'méxico', 'morelos', 'hidalgo','tlaxcala']

cases_per_state = dict()

for state in df.State.unique():

    key = state.lower()

    if key == 'ciudad de méxico':

        key = 'distrito federal'

    if key == 'queretaro':

        key = 'querétaro'

    cases_per_state[key] = len(df[df['State'] == state])



data['CPSTATE']= data['NAME'].map(cases_per_state)

data['coords'] = data['geometry'].apply(lambda x: x.representative_point().coords[:])

data['coords'] = [coords[0] for coords in data['coords']]

center_mx = data.loc[data['NAME'].isin(center_states)]



data['CPSTATE'] = data['CPSTATE']*8

center_mx = data.loc[data['NAME'].isin(center_states)]

# ----------------

# Complete dataset

# ----------------

path = '/kaggle/input/covid19-mx/covid-19_general_MX.csv'

df = pd.read_csv(path)

df['DIAS_INCUBACION'] = pd.to_datetime(df['FECHA_INGRESO'])-pd.to_datetime(df['FECHA_SINTOMAS'])

df['DIAS_INCUBACION'] = df['DIAS_INCUBACION'].dt.days

positive_ip_g14 = df.loc[(df['DIAS_INCUBACION'] > 14) & (df['RESULTADO'] == 1)]
df.head()
fig, ax = plt.subplots(figsize=(4,1))

ax.text(0.0, 1.0, 'The dataset currently contains a total of {} individuals'.format(len(df)), dict(size=25))

ax.text(0.0, 0.5, 'from which {} individuals have died.'.format(len(df.loc[(df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])), dict(size=25))

ax.text(0.0, 0.0, 'There\'s no information on recovered patients.'.format(len(df.loc[(df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])), dict(size=20))

ax.axis('off')

plt.show()
covid_positive = df.loc[df['RESULTADO'] == 1]

deads_positive = df.loc[(df['RESULTADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

alive_intubated = df.loc[(df['RESULTADO'] == 1) & ((df['FECHA_DEF'] == '9999-99-99') | (df.FECHA_DEF.isnull())) & (df['INTUBADO'] == 1)]

icu_alive = df.loc[(df['RESULTADO'] == 1) & ((df['FECHA_DEF'] == '9999-99-99') | (df.FECHA_DEF.isnull())) & (df['UCI'] == 1)]



cpcounts = len(covid_positive)

dpcounts = len(deads_positive)

aicounts = len(alive_intubated)

iccounts = len(icu_alive)



colors = ['#b00c00', '#edad5f', '#d69e04', '#b5d902', '#63ba00', '#05b08e', '#128ba6', '#5f0da6', '#b30bb0', '#c41484', '#a1183d', '#3859eb', '#4da1bf', '#6bcfb6']



sizes = np.array([cpcounts-dpcounts-aicounts-iccounts, dpcounts, iccounts,aicounts])

# Plot

fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('COVID-19 confirmed cases status distribution')

patches, texts = ax1.pie(sizes,colors=colors, startangle=90, shadow=True, explode=(0.0,0.1,0.1,0.1),

                         wedgeprops={'linewidth': 2,"edgecolor":"#303030", 'linestyle': 'solid', 'antialiased': True})



porcent = 100.*sizes/sizes.sum()

tags = ['Positive', 'Deceased', 'Alive in intensive care', 'Alive Intubated']

labels = ['{0} - {1:0.2f}% = {2:0.0f}'.format(tags[x],porcent[x],sizes[x]) for x in range(len(tags))]



ax1.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=15)

fig.tight_layout()
fig, ax = plt.subplots(figsize=(4,1))

ax.text(0.0, 1.0, 'Without speculation we have a mortality rate of {0:.2f}%'.format(porcent[1]), dict(size=22))

ax.text(0.0, 0.0, 'and {0:.2f}% of confirmed COVID-19 patients find themselves in intensive care units.'.format(porcent[2]+porcent[3]), dict(size=22))

ax.axis('off')

plt.show()
deads_negative = df.loc[(df['RESULTADO'] == 2) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

deads_pending = df.loc[(df['RESULTADO'] == 3) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]



dpcounts

dncounts = len(deads_negative)

dpecounts = len(deads_pending)



sizes = np.array([dpcounts,dncounts,dpecounts])

# Plot

fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('Deceased and COVID-19 test result correlation')

patches, texts = ax1.pie(sizes,colors=colors, startangle=90, shadow=True, explode=(0,0,0.2),

                         wedgeprops={'linewidth': 2,"edgecolor":"#303030", 'linestyle': 'solid', 'antialiased': True})



labels = ['{0} - {1:.2f}% = {2}'.format(i,100*j/sum(sizes),j) for i,j in zip(list(['Positive', 'Negative', 'Pending']), sizes)]

sort_legend = False

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, sizes),

                                          key=lambda x: x[2],

                                          reverse=True))





ax1.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=15, title="Test result")

fig.tight_layout()
positive_death_ninp = df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

positive_death_ip = df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

positive_death_ointubeted = df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

positive_death_opneumonia = df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]



negative_death_ninp = df.loc[(df['RESULTADO'] == 2) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

negative_death_ip = df.loc[(df['RESULTADO'] == 2) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

negative_death_ointubeted = df.loc[(df['RESULTADO'] == 2) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

negative_death_opneumonia = df.loc[(df['RESULTADO'] == 2) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]



inconclusive_death_ninp = df.loc[(df['RESULTADO'] == 3) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

inconclusive_death_ip = df.loc[(df['RESULTADO'] == 3) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

inconclusive_death_ointubeted = df.loc[(df['RESULTADO'] == 3) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]

inconclusive_death_opneumonia = df.loc[(df['RESULTADO'] == 3) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]



nr_ninp = [len(positive_death_ninp), len(negative_death_ninp), len(inconclusive_death_ninp)]

nr_ip = [len(positive_death_ip), len(negative_death_ip), len(inconclusive_death_ip)]

nr_intubated = [len(positive_death_ointubeted), len(negative_death_ointubeted), len(inconclusive_death_ointubeted)]

nr_pneumonia = [len(positive_death_opneumonia), len(negative_death_opneumonia), len(inconclusive_death_opneumonia)]



tags = ['Positive', 'Negative', 'Pending']

tags_legend = ['No intubation nor pneumonia', 'Intubation & pneumonia', 'Intubation only','Pneumonia only']



fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('Deaths related to pneumonia and intubation distributed by test result')

x = np.arange(len(tags))

ax1.bar(x, nr_ninp, width=0.4, color=colors[0], align='center')

ax1.bar(x, nr_ip, width=0.4, color=colors[1], align='center', bottom=nr_ninp)

ax1.bar(x, nr_intubated, width=0.4, color=colors[2], align='center', bottom=np.array(nr_ip)+np.array(nr_ninp))

ax1.bar(x, nr_pneumonia, width=0.4, color=colors[3], align='center', bottom=np.array(nr_intubated)+np.array(nr_ninp)+np.array(nr_ip))

ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], label='{0}'.format(tags_legend[x])) for x in range(4)], 

                     loc='best', fancybox=True, shadow=True, title="Death related to:")

plt.xticks(x, tags)

plt.show()
from collections import OrderedDict

pcv_ns = len(df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] != 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])

pcv_inpn = len(df.loc[(df['RESULTADO'] == 1) & (df['INTUBADO'] == 1) & (df['NEUMONIA'] == 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])

pcv_oint = len(df.loc[(df['RESULTADO'] == 1) & (df['INTUBADO'] == 1) & (df['NEUMONIA'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])

pcv_opne = len(df.loc[(df['RESULTADO'] == 1) & (df['NEUMONIA'] == 1) & (df['INTUBADO'] != 1) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())])

d = {'Non specified':pcv_ns,

        'Pneumonia and intubation':pcv_inpn,

        'Intubation without pneumonia':pcv_oint,

        'Pneumonia without intubation':pcv_opne

       }

d = OrderedDict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))

tags = list(d.keys())

sizes = np.array(list(d.values()))

# Plot

fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('Confirmed COVID-19 deceased cause of death')

patches, texts = ax1.pie(sizes,colors=colors, startangle=90, shadow=True, explode=(0.2,0,0,0.0),

                         wedgeprops={'linewidth': 2,"edgecolor":"#303030", 'linestyle': 'solid', 'antialiased': True})



labels = ['{0} - {1:.2f}% = {2}'.format(i,100*j/sum(sizes),j) for i,j in zip(list(tags), sizes)]

sort_legend = False

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, sizes),

                                          key=lambda x: x[2],

                                          reverse=True))





ax1.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=15, title="Death associated with")

fig.tight_layout()
inds = d['Pneumonia and intubation']+d['Intubation without pneumonia']+d['Pneumonia without intubation']

porcent = 100*inds/sum(d.values())

fig, ax = plt.subplots(figsize=(4,1))

ax.text(0.0, 1.0, 'From the confirmed COVID-19 cases {0:.2f}% were linked'.format(porcent), dict(size=22))

ax.text(0.0, 0.5, 'to either pneumonia, being intubated, or both', dict(size=22))

ax.text(0.0, 0.0, 'Representing a total of {} individuals from the total {} deceased.'.format(inds, dpcounts),dict(size=22)) 

ax.axis('off')

plt.show()
fig, ax = plt.subplots(figsize=(4,1))

ax.text(0.0, 1.0, 'Using the previously discussed information we can assume that', dict(size=22))

ax.text(0.0, 0.5, '{0:.2f}% of the deaths related with a negative or pending'.format(porcent), dict(size=22))

ax.text(0.0, 0.0, 'test are related somehow to COVID-19.'.format(inds, dpcounts),dict(size=22)) 

ax.axis('off')

plt.show()
path_sectors = '/kaggle/input/covid19-mx/SECTOR.csv'

df_sector = pd.read_csv(path_sectors)

df_sector['TOTAL_M'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]) for x in list(df_sector['CLAVE'])]

df_sector['TOTAL_MP'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull()) & (df['RESULTADO'] == 1)]) for x in list(df_sector['CLAVE'])]

df_sector['TOTAL_P'] = [len(df.loc[(df['SECTOR'] == x) & (df['RESULTADO'] == 1)]) for x in list(df_sector['CLAVE'])]

df_sector['TOTAL_MP_PN'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull()) & ((df['RESULTADO'] == 1) | (df['NEUMONIA'] == 1) | (df['INTUBADO'] == 1))]) for x in list(df_sector['CLAVE'])]

df_sector['PP_PM_PN'] = (100*df_sector['TOTAL_MP_PN'])/df_sector['TOTAL_P']

df_sector['PP_TM'] = (100*df_sector['TOTAL_M'])/df_sector['TOTAL_P']

df_sector['PP_PM'] = (100*df_sector['TOTAL_MP'])/df_sector['TOTAL_P']

df_sector = df_sector.sort_values('TOTAL_P', ascending=False)

df_sector = df_sector.fillna(0)

df_sector = df_sector.sort_values('TOTAL_P', ascending=True)

df_sector = df_sector.fillna(0)



fig, ax1 = plt.subplots(figsize=(12,8))

ax1.set_title('Confirmed vs inferred COVID-19 deaths')

ax1.barh(df_sector['DESCRIPCIÓN'], df_sector['TOTAL_MP_PN']*(porcent/100), align='center', label="Inferred deaths (pneumonia and intubated cases)")

ax1.barh(df_sector['DESCRIPCIÓN'], df_sector['TOTAL_MP'], align='center', label="Confirmed COVID-19 deaths")

for i, v in enumerate(df_sector['TOTAL_MP_PN']*(porcent/100)):

    positive = list(df_sector['TOTAL_MP'])[i]

    suposed = v

    if positive > 0 or suposed > 0:

        ax1.text(v + 3, i - 0.25, '{0} vs {1:0.0f}'.format(positive, suposed))



ax1.legend()

ax1.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)

ax1.yaxis.set_ticks_position('left')

ax1.xaxis.set_ticks_position('bottom')
colors = ['#b00c00', '#edad5f', '#d69e04', '#b5d902', '#63ba00', '#05b08e', '#128ba6', '#5f0da6', '#b30bb0', '#c41484', '#a1183d', '#3859eb', '#4da1bf', '#6bcfb6']

df_sector['TOTAL'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull()) & (df['RESULTADO'] == 1)]) for x in list(df_sector['CLAVE'])]

df_sector = df_sector.sort_values('TOTAL', ascending=False)



sizes = list(df_sector['TOTAL'])



# Plot

fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('Deceased COVID-19 confirmed cases distribution by healthcare institution')

patches, texts = ax1.pie(sizes,colors=colors, startangle=90, shadow=True,

                         wedgeprops={'linewidth': 2,"edgecolor":"#303030", 'linestyle': 'solid', 'antialiased': True})



labels = ['{0} - {1:.2f}% = {2}'.format(i,100*j/sum(sizes),j) for i,j in zip(list(df_sector['DESCRIPCIÓN']), sizes)]

sort_legend = False

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, sizes),

                                          key=lambda x: x[2],

                                          reverse=True))





ax1.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=15)

fig.tight_layout()
df_sector['TOTAL'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]) for x in list(df_sector['CLAVE'])]

df_sector = df_sector.sort_values('TOTAL', ascending=False)



sizes = list(df_sector['TOTAL'])



# Plot

fig, ax1 = plt.subplots(figsize=(20,10))

ax1.set_title('Total deceased distribution by healthcare institution')

patches, texts = ax1.pie(sizes,colors=colors, startangle=90, shadow=True,

                         wedgeprops={'linewidth': 2,"edgecolor":"#303030", 'linestyle': 'solid', 'antialiased': True})



labels = ['{0} - {1:.2f}% = {2}'.format(i,100*j/sum(sizes),j) for i,j in zip(list(df_sector['DESCRIPCIÓN']), sizes)]



sort_legend = True

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, sizes),

                                          key=lambda x: x[2],

                                          reverse=True))





ax1.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.), fontsize=15)

fig.tight_layout()
df_sector['TOTAL_M'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull())]) for x in list(df_sector['CLAVE'])]

df_sector['TOTAL_MP'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull()) & (df['RESULTADO'] == 1)]) for x in list(df_sector['CLAVE'])]

df_sector['TOTAL_P'] = [len(df.loc[(df['SECTOR'] == x) & (df['RESULTADO'] == 1)]) for x in list(df_sector['CLAVE'])]

df_sector = df_sector.sort_values('TOTAL_P', ascending=False)

df_sector = df_sector.fillna(0)

fig, ax1 = plt.subplots(figsize=(20,8))

ax1.set_title('COVID-19 confirmed cases and deceases per Healthcare institution')

x = np.arange(len(df_sector['DESCRIPCIÓN']))

w=0.3

total_p = ax1.bar(x, df_sector['TOTAL_P'], width=w, color='#edad5f', align='center')

for i, bar in enumerate(total_p):

    bar.set_color(colors[i])

total_mp = ax1.bar(x + w, df_sector['TOTAL_MP'], width=w, align='center')

legend1 = ax1.legend([total_mp],['Deceased'])

ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], label='{0} - {1:0.2f}%'.format(list(df_sector['DESCRIPCIÓN'])[x],list(df_sector['PP_PM'])[x])) for x in range(len(df_sector['DESCRIPCIÓN']))], 

                     loc='best',bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True, title="Mortality Rate:")

plt.xticks(x + w /2, df_sector['DESCRIPCIÓN'], rotation='vertical')

plt.gca().add_artist(legend1)

plt.show()
df_sector['TOTAL_MP_PN'] = [len(df.loc[(df['SECTOR'] == x) & (df['FECHA_DEF'] != '9999-99-99') & (df.FECHA_DEF.notnull()) & ((df['RESULTADO'] == 1) | (df['NEUMONIA'] == 1) | (df['INTUBADO'] == 1))]) for x in list(df_sector['CLAVE'])]

df_sector['PP_PM_PN'] = (100*df_sector['TOTAL_MP_PN'])/df_sector['TOTAL_P']

df_sector = df_sector.sort_values('TOTAL_P', ascending=False)

df_sector = df_sector.fillna(0)



fig, ax1 = plt.subplots(figsize=(20,8))

ax1.set_title('COVID-19 confirmed cases and deceases per Healthcare institution')

x = np.arange(len(df_sector['DESCRIPCIÓN']))

w=0.3

total_p = ax1.bar(x, df_sector['TOTAL_P'], width=w, color='#edad5f', align='center')

for i, bar in enumerate(total_p):

    bar.set_color(colors[i])

total_mp = ax1.bar(x + w, df_sector['TOTAL_MP_PN']*(porcent/100), width=w, align='center')

legend1 = ax1.legend([total_mp],['Deceased'])

ax1.legend(handles=[matplotlib.patches.Patch(facecolor=colors[x], label='{0} - {1:0.2f}%'.format(list(df_sector['DESCRIPCIÓN'])[x],list(df_sector['PP_PM_PN'])[x])) for x in range(len(df_sector['DESCRIPCIÓN']))], 

                     loc='best',bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True, title="Mortality Rate:")

plt.xticks(x + w /2, df_sector['DESCRIPCIÓN'], rotation='vertical')

plt.gca().add_artist(legend1)

plt.show()
deltas = mx_covid['Confirmed Cases']

deltas = [deltas[x] if x == 0 else deltas[x]-deltas[x-1] for x in range(len(deltas))]

fig, ax = plt.subplots(figsize=(20,8))

ax.set_title('Confirmed cases delta')

ax.plot(pd.to_datetime(mx_covid['Dates']), deltas,color='orange')

ax.bar(pd.to_datetime(mx_covid['Dates']), deltas)

for line, name in zip(ax.lines, ['MAX new cases']):

    y = max(line.get_ydata())

    ax.annotate('{} {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points",

                size=14, va="center")

fig.autofmt_xdate()
fig, ax = plt.subplots(figsize=(20,10))

ax.set_title('Speculated vs Official COVID-19 cases')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*8, label='8')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*10, label='10')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*12, label='12')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases'], label='')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Estimation scaling factor", fancybox=True)

for line, name in zip(ax.lines, ['with scaling factor of 8', 'with scaling factor of 10', 'with scaling factor of 12', 'with no scaling factor']):

    y = line.get_ydata()[-1]

    ax.annotate('{} {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points",

                size=14, va="center")
def simulate_infections(incubation_days, scaling_factor):

    added_infected = dict()

    for i, x in enumerate(mx_covid['Confirmed Cases']):

        if added_infected.get(i) is None:

            added_infected[i] = x

        else:

            added_infected[i] += x

        if added_infected.get(i+incubation_days) is None:

            added_infected[i+incubation_days] = x*scaling_factor 

        else:

            added_infected[i+incubation_days] += x*scaling_factor 

    xl = []

    for i in range(len(mx_covid['Confirmed Cases'])):

        xl.append(added_infected[i])

    return xl
fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('Different incubation periods with 8 people infected per case confirmed')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*8, label='estimated x8')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(3, 8), label='3 day incubation')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(4, 8), label='4 day incubation')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(5, 8), label='5 day incubation')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases'], label='official')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Legend", fancybox=True)

for line, name in zip(ax.lines, ['scaling factor of 8', '3 days of incubation', '4 days of incubation', '5 days of incubation']):

    y = line.get_ydata()[-1]

    ax.annotate('{} with {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points", size=14, va="center")

fig.autofmt_xdate()

fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('Infected per case confirmed with 5 day incubation')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*8, label='estimated x8')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(5, 10), label='10 infected pcc')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(5, 12), label='12 infected pcc')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(5, 14), label='14 infected ppc')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Legend", fancybox=True)

for line, name in zip(ax.lines, ['scaling factor of 8', '10 infections per case confirmed', '12 infections per case confirmed', '14 infections per case confirmed']):

    y = line.get_ydata()[-1]

    ax.annotate('{} with {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points", size=14, va="center")

fig.autofmt_xdate()
G=nx.Graph()

G.add_nodes_from(['Confirmed Case',1,2,3,4,5,6,7,8,9,10,11,12])

G.add_edges_from([('Confirmed Case',1), ('Confirmed Case',2), ('Confirmed Case',3), (1,4), (1,5), (1,6), (2,7), (2,8), (3,9), (3,10), (3, 11), (3,12)])



color_nodes = []

for node in G:

    if node == 'Confirmed Case':

        color_nodes.append('#f55333')

        continue

    if node < 4:

        color_nodes.append('#f58733')

        continue

    color_nodes.append('#f5d533')



plt.figure(1,figsize=(12,8)) 

nx.draw(G, node_size = 2000, node_color=color_nodes, with_labels = True)

plt.show()
fig, ax = plt.subplots(figsize=(20,8))

ax.set_title('Estimations, infections per case confirmed & official information')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*8, label='estimated scaling factor of 8')

ax.plot(pd.to_datetime(mx_covid['Dates']), simulate_infections(5, 12), label='12 infected pcc')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases'], label='official')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Legend", fancybox=True)

for line, name in zip(ax.lines, ['with scaling factor of 8', 'with 12 infected per case confirmed', 'with official numbers']):

    y = line.get_ydata()[-1]

    ax.annotate('{} {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points",

                size=14, va="center")

fig.autofmt_xdate()
path = '/kaggle/input/mxstatesdataset/Mexico_States.shp'

data = gpd.read_file(path)

data['NAME'] = data['NAME'].str.lower()



df = pd.read_csv('/kaggle/input/covid19-mx/casos_confirmados.csv')

df = df.dropna()

df.head()

center_states = ['distrito federal', 'querétaro', 'puebla', 'méxico', 'morelos', 'hidalgo','tlaxcala']

cases_per_state = dict()

for state in df.State.unique():

    key = state.lower()

    if key == 'ciudad de méxico':

        key = 'distrito federal'

    if key == 'queretaro':

        key = 'querétaro'

    cases_per_state[key] = len(df[df['State'] == state])



data['CPSTATE']= data['NAME'].map(cases_per_state)

data['coords'] = data['geometry'].apply(lambda x: x.representative_point().coords[:])

data['coords'] = [coords[0] for coords in data['coords']]

center_mx = data.loc[data['NAME'].isin(center_states)]



data['CPSTATE'] = data['CPSTATE']*8

center_mx = data.loc[data['NAME'].isin(center_states)]



fig, ax1 = plt.subplots(figsize=(25,15))



left, bottom, width, height = [0.5, 0.55, 0.25, 0.25]

ax2 = fig.add_axes([left, bottom, width, height])



data.plot(ax=ax1, column='CPSTATE', cmap='Reds',edgecolor="black", legend=True, legend_kwds={'label': "Official confirmed COVID-19 cases", 'shrink':0.5})

for idx, row in data.iterrows():

    if row['NAME'] not in center_states:

        ax1.annotate(s=row['CPSTATE'], xy=row['coords'],horizontalalignment='center')

        

center_mx.plot(ax=ax2, column='CPSTATE', cmap='Reds',edgecolor="black", legend=False)

for idx, row in center_mx.iterrows():

    ax2.annotate(s=row['CPSTATE'], xy=row['coords'],horizontalalignment='center')

    

ax1.axis('off')

ax2.axis('off')

ax1.legend(fontsize=8)

plt.show()

plt.close()
def simulate_infections_predict(incubation_days, scaling_factor, predict_days=10, base=mx_covid['Confirmed Cases']):

    added_infected = dict()

    last_day = 1

    for i, x in enumerate(base):

        if added_infected.get(i) is None:

            added_infected[i] = x

        else:

            added_infected[i] += x

        if added_infected.get(i+incubation_days) is None:

            added_infected[i+incubation_days] = x*scaling_factor 

        else:

            added_infected[i+incubation_days] += x*scaling_factor 

        last_day = i+incubation_days

    for day in range(predict_days):

        day_pinc = last_day-(incubation_days-1)

        prev_infected = added_infected[day_pinc]/8

        added_infected[last_day+1] = int(prev_infected*scaling_factor)

        last_day+=1

    return [added_infected[x] for x in range(len(added_infected))]
pred_dates=get_date_list(datetime.datetime.today() + datetime.timedelta(days=15),total=len(mx_confirmed_cases)+15)

pred_per_day=np.array(simulate_infections_predict(5, 12))

fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('Prediction for the next {} days estimations'.format(len(pred_per_day)-len(mx_covid['Confirmed Cases'])))

ax.plot(pd.to_datetime(pred_dates), pred_per_day, label='Estimated prediction')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases']*8, label='Current estimations x8')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Legend", fancybox=True)

for line, name in zip(ax.lines, ['infected of COVID-19 on {}'.format(pred_dates[-1])]):

    y = line.get_ydata()[-1]

    ax.annotate('{} {}'.format(y, name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points",

                size=14, va="center")

fig.autofmt_xdate()
fig, ax = plt.subplots(figsize=(12,8))

ax.set_title('Prediction for the next {} days official'.format(len(pred_per_day)-len(mx_covid['Confirmed Cases'])))

ax.plot(pd.to_datetime(pred_dates), pred_per_day/8, label='Prediction of official cases')

ax.plot(pd.to_datetime(mx_covid['Dates']), mx_covid['Confirmed Cases'], label='Official cases')

ax.legend(loc='upper left', shadow=True, bbox_to_anchor=[0, 1], ncol=2, title="Legend", fancybox=True)

for line, name in zip(ax.lines, ['official cases of COVID-19 on {}'.format(pred_dates[-1])]):

    y = line.get_ydata()[-1]

    ax.annotate('{} {}'.format(int(y), name), xy=(1,y), xytext=(6,0), color=line.get_color(), 

                xycoords = ax.get_yaxis_transform(), textcoords="offset points",

                size=14, va="center")

fig.autofmt_xdate()