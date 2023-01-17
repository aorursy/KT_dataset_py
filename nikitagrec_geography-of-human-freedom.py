import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data18 = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')

data18.shape
data18.head(3)
data18.info()
from mpl_toolkits.basemap import Basemap

concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

concap.head(3)
data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         data18,left_on='CountryName',right_on='countries')
def mapWorld():

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full['hf_score'].values

    #a_2 = data_full['Economy (GDP per Capita)'].values

    #300*a_2

    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='hot', alpha=1)

    

    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)

    cbar = m.colorbar()

    cbar.set_label('Human Freedom',fontsize=30)

    #plt.clim(20000, 100000)

    plt.title("Human Freedom (score)", fontsize=30)

    plt.show()

plt.figure(figsize=(30,30))

mapWorld()
lst = ['pf_rol_procedural','pf_rol','pf_score','ef_legal','ef_trade','ef_score','hf_score']



def reg(x):

    if x=='Middle East & North Africa':

        res = 'Mid East & Nor Afr'

    elif x=='Latin America & the Caribbean':

        res = 'Lat Amer & Car'

    elif x=='Caucasus & Central Asia':

        res = 'Cauc & Cen Asia'

    elif x=='Sub-Saharan Africa':

            res = 'Sub-Sah Africa'

    else:

        res=x

    return res

data_bx = data18

data_bx['region'] = data_bx.region.apply(reg)



plt.figure(figsize=(30,10))

sns.set(style="white",font_scale=1.5)

sns.boxplot(x='region',y='hf_score',data=data_bx);

sns.swarmplot(x='region',y='hf_score',data=data_bx,color=".25");
data18.corr()[abs(data18.corr())>0.72]['hf_score'].dropna()[['pf_rol_procedural', 'pf_rol_civil', 'pf_rol_criminal', 'pf_rol',\

       'pf_ss', 'pf_expression_influence',\

       'pf_expression_control', 'pf_expression', 'pf_score',\

        'ef_legal', 'ef_trade', 'ef_score']]
def mapWorld(col1,size2,title3,label4,metr=100,colmap='hot'):

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    

    #m.drawmapboundary(fill_color='#FFFFFF')

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full[col1].values

    if size2:

        a_2 = data_full[size2].values

    else: a_2 = 1

    #300*a_2

    m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,linewidth=1,edgecolors='black',cmap=colmap, alpha=1)

    

    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)

    cbar = m.colorbar()

    cbar.set_label(label4,fontsize=30)

    #plt.clim(20000, 100000)

    plt.title(title3, fontsize=30)

    plt.show()

plt.figure(figsize=(30,30))

mapWorld('hf_score','pf_score',"Human Freedom (score)", 'Human Freedom')
plt.figure(figsize=(30,30))

mapWorld('ef_score','ef_trade',"Economic Freedom (score)", 'Economic Freedom',metr=100,colmap='viridis')
plt.figure(figsize=(30,30))

mapWorld(col1='pf_ss_disappearances_injuries',size2=False,title3="Terrorism fatalities",\

         label4='Terrorism fatalities',metr=700,colmap='viridis')
plt.figure(figsize=(30,10))

sns.set(style="white",font_scale=1.5)

sns.boxplot(x='region',y='hf_score',data=data_bx);

sns.swarmplot(x='region',y='pf_ss_disappearances_injuries',data=data_bx,color=".25");
plt.figure(figsize=(30,30))

mapWorld(col1='pf_rol_procedural',size2='pf_rol',title3="Procedural justice",\

         label4='Procedural justice',metr=200,colmap='viridis')
lstt = ['ef_regulation_labor','ef_regulation_business_adm','ef_regulation_business_bureaucracy','ef_regulation_business_start',\

'ef_regulation_business_bribes','ef_regulation_business_licensing','ef_regulation_business_compliance','ef_regulation_business',\

'ef_regulation']
data18.corr()[lstt][-3:]
idx=1

def ff(x):

    return dict_val[x]

plt.figure(figsize=(20,50))

qqq = ['ef_regulation_business_bureaucracy','ef_regulation_business_start','ef_regulation_business_bribes',\

'ef_regulation_business','ef_regulation']

my_data = data_full[qqq+['region']].dropna()

my_data.columns = ['Bureaucracy costs','Starting a business','Extra payments','Business regulations', \

            'Regulation','region']

qq_1 = ['Bureaucracy costs','Starting a business','Extra payments','Business regulations','Regulation','region']

for position_name, features in my_data.groupby(my_data['region'])[qq_1].median().iterrows():

    feat = dict(features)

    

    categs=feat.keys()

    N = len(categs)



    values = list(feat.values())

    values += values[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    ax = plt.subplot(9, 3, idx, polar=True)



    plt.xticks(angles[:-1], categs, color='grey', size=15)

    ax.set_rlabel_position(0)

    plt.yticks([2,5,10], ["2","5","10"], color="grey", size=15)

    plt.ylim(0,10)

    

    plt.subplots_adjust(hspace = 0.5)

    ax.plot(angles, values, linewidth=3, linestyle='solid')

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=15, y=1.1)

    

    idx += 1
plt.figure(figsize=(30,30))

mapWorld(col1='pf_ss_women_missing',size2='pf_ss_women',title3="Missing women",\

         label4='Missing women',metr=100,colmap='viridis')
def ff(x):

    return dict_val[x]

plt.figure(figsize=(20,50))

qqq = ['pf_ss_women_missing','pf_ss_women_inheritance_widows','pf_ss_women_inheritance_daughters',\

       'pf_ss_women_inheritance','pf_ss_women','pf_ss_women_fgm']

my_data = data_full[qqq+['region']].dropna()

my_data.columns = ['Missing women','Inheritance rights for widows','Inheritance rights for daughters',\

'Inheritance',"Women's security",'Female genital mutilation','region']

qq_1 = list(my_data.columns)

def spyder_plot(qq_1):

 idx=1

 for position_name, features in my_data.groupby(my_data['region'])[qq_1].median().iterrows():

    feat = dict(features)

    

    categs=feat.keys()

    N = len(categs)



    values = list(feat.values())

    values += values[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]

    ax = plt.subplot(9, 3, idx, polar=True)

    plt.xticks(angles[:-1], categs, color='grey', size=15)

    ax.set_rlabel_position(0)

    plt.yticks([2,5,10], ["2","5","10"], color="grey", size=15)

    plt.ylim(0,10)

    

    plt.subplots_adjust(hspace = 0.5)

    ax.plot(angles, values, linewidth=3, linestyle='solid')

    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(position_name, size=15, y=1.1)

    idx += 1

spyder_plot(qq_1)
plt.figure(figsize=(30,30))

mapWorld(col1='pf_association_association',size2=False,title3="Freedom of association",\

         label4='Freedom of association',metr=500,colmap='viridis')
idx=1

def ff(x):

    return dict_val[x]

plt.figure(figsize=(20,50))

qqq = ['pf_association_association','pf_association_assembly','pf_association_political_establish',\

'pf_association_political_operate','pf_association_political']

my_data = data_full[qqq+['region']].dropna()

spyder_plot(qqq)
plt.figure(figsize=(30,30))

mapWorld(col1='ef_government',size2=False,title3="",\

         label4='Size of government ',metr=500,colmap='viridis')
plt.figure(figsize=(20,50))

qqq = ['ef_government_consumption','ef_government_transfers','ef_government_enterprises',\

'ef_government_tax_payroll','ef_government']

my_data = data_full[qqq+['region']].dropna()

spyder_plot(qqq)
plt.figure(figsize=(30,30))

mapWorld(col1='ef_money_inflation',size2=False,title3="Inflation: most recent year",\

         label4='',metr=500,colmap='viridis')
plt.figure(figsize=(20,50))

qqq = ['ef_money_growth','ef_money_sd','ef_money_inflation',\

'ef_money_currency','ef_money']

my_data = data_full[qqq+['region']].dropna()

spyder_plot(qqq)