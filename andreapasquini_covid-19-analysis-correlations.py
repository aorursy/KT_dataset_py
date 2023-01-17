# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

import warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML, display

def tables_side(titolo, tabs, tabslabel):
    return HTML( '<table valign="top"><caption style="background-color:#F1F2F3;"><h1 align="center">'+titolo+'</h1></caption>'+
                 '<tr style="background-color:#D9D9D9;" >'+''.join(['<td><h2 align="center">'+tlabel+ '</h2></td>' for tlabel in tabslabel]) +'</tr>'+
                 '<tr style="background-color:#D9D9D9;" valign="top">'+''.join(['<td  valign="top">'+table._repr_html_() + '</td>' for table in tabs]) +
                 '</tr></table>'
               )

%matplotlib inline
df = pd.read_csv('../input/covidcorrelations/PopAgeLevelProv2017.csv')
df.head(3)
# Select only global sex and fix all issue regarding the age (converting strings in integers)
df = df[df.Sesso=='totale']
df = df[df.STATCIV2==99]
df = df.drop(df[df.ETA1=='TOTAL'].index)
df.ETA1 = df.ETA1.apply(lambda x: (x.replace('_GE','')))
df.ETA1 = df.ETA1.apply(lambda x: (x.replace('Y','')))
df.ETA1 = df.ETA1.apply(lambda x: int(x))

# I create a column to be able to calculate the average age of the population with the weighted average
df['ExP'] = df.apply(lambda x: x.Value if x.ETA1==0 else x.ETA1*x.Value, axis=1)


print('I create a column to be able to calculate the average age of the population with the weighted average, and I called it ExP')
tables_side(titolo='Milano',
            tabs=[df.loc[df.Territorio=='Milano',['Value','ETA1', 'ExP']].head(20)],
            tabslabel=['People by age'])
df
city = ['Milano', 'Roma', 'Bergamo', 'Bologna', 'Napoli','Firenze','Torino','Genova']
ages = np.arange(0, 110, 5)

plt.figure( figsize=(25,10))
for i,citta in zip(np.arange(len(city)), city):
    y = df[df.Territorio==city[i]].Value/df.loc[df.Territorio==city[i],'Value'].sum()*100
    plt.plot(df[df.Territorio==city[i]].ETA1, y, label = city[i])

    
plt.xticks(ages)    

plt.annotate("What's happen here?", xy=(67, 0.85),
             xycoords='data',
             size = 15,
             xytext=(50, 0.1),
             textcoords='data',
             arrowprops=dict(arrowstyle= '-|>',
                             color='r',
                             lw=5,
                             ls='-')
           )

plt.plot(73, 1.1, 'o', ms=130 * 2, mec='b', mfc='none', mew=2)
    
plt.grid(color='#d4d4d4')
plt.xlabel('Date', size=15)
plt.ylabel('% of population', size=15)

plt.text(x = -5, y = (y.max()+y.max()*0.15), s = "Percentage of people grouped by age and provinces", fontsize = 23, weight = 'bold', alpha = .75)
plt.text(x = -5, y = (y.max()+y.max()*0.06), s = "For Milan, Rome, Bergamo, Bologna, Naples, Florence, Turin and Genoa\nthe largest Italian cities and distributed throughout the Italian territory.", fontsize = 16, alpha = .75)
plt.text(x = -5, y = -(y.max()*0.2),s = 'Source: https://raw.githubusercontent.com/andreapas79/COVID-19/master/Data/demographic_data/DataforKaggle/PopAgeLevelProv2017.csv', fontsize = 10)


plt.legend()
plt.show()
years = np.arange(1907, 2017, 1)[::-1]
ages = np.arange(0, 110, 1)

plt.figure( figsize=(25,10))
for i,citta in zip(np.arange(len(city)), city):
    plt.plot(df[df.Territorio==city[i]].ETA1, df[df.Territorio==city[i]].Value/df.loc[df.Territorio==city[i],'Value'].sum()*100, label = city[i])

    
plt.xticks(ages, years, rotation=90,size=10)    

plt.annotate('Too many childrens. People stop to have kids ', xy=(68, 1.35),
             xycoords='data',
             size = 15,
             xytext=(78, 1.7),
             textcoords='data',
             arrowprops=dict(arrowstyle= '-|>',
                             color='r',
                             lw=5,
                             ls='-')
           )
    
plt.grid(color='#d4d4d4')
plt.xlabel('Date',size=15)
plt.ylabel('% of population', size=15)

plt.legend(loc='center right')


plt.axvline(x=71, ymin=0, ls='--')
plt.text(69.5,0.1,'1945 - end WW2',rotation=90, size=15)
plt.axvline(x=76, ymin=0, ls='--')
plt.text(74.5,0.1,'1940 - starting WW2',rotation=90, size=15)

plt.axvline(x=59, ymin=0, ls='--')
plt.text(57.5,0.1,'1957-58 Asiatic-flu H2N2',rotation=90, size=15)

plt.axvline(x=48, ymin=0, ls='--')
plt.text(46.5,0.1,'1968 HongKong-flu A/H3N2',rotation=90, size=15)

plt.axvline(x=39, ymin=0, ls='--')
plt.text(37.5,0.1,'1977 Russian-flu H1N1',rotation=90, size=15)

plt.axvline(x=99, ymin=0, ls='--')
plt.text(97.5,0.1,'1918 Spanish flu H1N1',rotation=90, size=15)

plt.text(x = -5, y = (y.max()+y.max()*0.15), s = "Percentage of people grouped by age and province during the history and the most important events", fontsize = 23, weight = 'bold', alpha = .75)
plt.text(x = -5, y = (y.max()+y.max()*0.06), s = "For Milan, Rome, Bergamo, Bologna, Naples, Florence, Turin and Genoa\nthe largest Italian cities and distributed throughout the Italian territory.", fontsize = 16, alpha = .75)
plt.text(x = -5, y = -(y.max()*0.2),s = 'Source: https://raw.githubusercontent.com/andreapas79/COVID-19/master/Data/demographic_data/DataforKaggle/PopAgeLevelProv2017.csv', fontsize = 10)


plt.show()
province = df.groupby(by=['Territorio','ITTER107']).sum().loc[:,('ExP','Value')]

province = province.reset_index()
province['MeanAge'] = 0
province['PercOver65'] = 0
for i in province.Territorio:
    province.loc[province.Territorio==i ,'MeanAge'] = round(df.loc[df.Territorio==i,'ExP'].sum()/province.loc[province.Territorio==i,'Value'],1)
    df_q = df.loc[df.Territorio==i]
    province.loc[province.Territorio==i ,'PercOver65'] = round(df_q.loc[df.ETA1>60].Value.sum()/df_q.Value.sum()*100,1 )
province.columns = ['Territorio', 'ITTER107', 'ExP', 'Popolazione', 'MeanAge', 'PercOver65']

tables_side(titolo='Italian Regions',
            tabs=[province],
            tabslabel=['Population, Mean age and %people older than 65yo'])
y = province.loc[province.ITTER107.apply(lambda x: len(x)==4),:].PercOver65
meanOver65= y.mean()
colors = ["grey", "red"]
mO = province.loc[province.ITTER107.apply(lambda x: len(x)==4),:].copy()
mO['Red'] = 0
mO.loc[(mO.Territorio.isin(['Lombardia', 'Emilia-Romagna', 'Piemonte']))&(mO.ITTER107.apply(lambda x: len(x)==4)), 'Red'] = 1

plt.figure(figsize=(20,10))
sns.barplot(data=mO[mO.ITTER107.apply(lambda x: len(x)==4)], y='PercOver65', x='Territorio', hue='Red', dodge=False, palette=colors)

plt.xlabel('Regions',size=15)
plt.ylabel('% of population over 65yo', size=15)

plt.axhline(xmax=22, y=meanOver65, ls='--', lw=5, color='salmon')
plt.text(0,meanOver65*1.025,'Mean value % over 65yo', size=15)
plt.legend('')
plt.xticks(rotation=45, size=10,ha='right')

plt.text(x = -0.5, y = (y.max()+y.max()*0.15), s = "Percentage of people over 65 years old", fontsize = 23, weight = 'bold', alpha = .75)
plt.text(x = -0.5, y = (y.max()+y.max()*0.10), s = "for each regions in relation to the national average.", fontsize = 16, alpha = .75)
plt.text(x = -0.5, y = -(y.max()*0.4),s = 'Source: https://raw.githubusercontent.com/andreapas79/COVID-19/master/Data/demographic_data/DataforKaggle/PopAgeLevelProv2017.csv', fontsize = 10)


plt.show()
province[province.ITTER107.apply(lambda x: ('ITC'== x)or('ITE'== x))]
tables_side(titolo='Provinces in North-West and Center of Italy ',
            tabs=[province[province.ITTER107.apply(lambda x: ('ITC' in x)or('ITE' in x))].head(10)],
            tabslabel=['Population, Mean age and %people older than 65yo'])
meanOver65= province.loc[province.ITTER107.apply(lambda x: len(x)==5),:].PercOver65.mean()
colors = ["grey", "red"]
mO = province.loc[(province.ITTER107.apply(lambda x: len(x)==5))&(province.ITTER107.apply(lambda x: ('ITC' in x)or('ITE' in x))),:].copy()
mO['Red'] = 0
mO.loc[(mO.ITTER107.apply(lambda x: 'ITC' in x))&(mO.ITTER107.apply(lambda x: len(x)==5)), 'Red'] = 1

plt.figure(figsize=(20,10))
sns.barplot(data=mO, y='PercOver65', x='Territorio', hue='Red', dodge=False, palette=colors)

plt.xlabel('Regions',size=15)
plt.ylabel('% of population over 65yo', size=15)

plt.axhline(xmax=22, y=meanOver65, ls='--', lw=5, color='salmon')
plt.text(0,meanOver65*1.025,'Mean value % over 65yo', size=15)
plt.legend('')
plt.xticks(rotation=45, size=10,ha='right')

plt.text(x = -0.5, y = (y.max()+y.max()*0.15), s = "Percentage of people over 65 years old", fontsize = 23, weight = 'bold', alpha = .75)
plt.text(x = -0.5, y = (y.max()+y.max()*0.10), s = "for Lombardy, Lazio provinces in relation to the national average.", fontsize = 16, alpha = .75)
plt.text(x = -0.5, y = -(y.max()*0.4),s = 'Source: https://raw.githubusercontent.com/andreapas79/COVID-19/master/Data/demographic_data/DataforKaggle/PopAgeLevelProv2017.csv', fontsize = 10)

plt.show()
df = pd.read_csv('../input/covidcorrelations/SurfaceKMQ.csv')
df = df[df['Tipo dato']=='superficie (kmq)']
df = df.groupby(by=['Territorio', 'ITTER107']).sum()['Value']
df = df.reset_index()

x = pd.read_csv('../input/covidcorrelations/PopDeath2017.csv')

trimestre = x.loc[(x.Territorio=='Lombardia')&(x.Sesso=='totale') &(x.TIPO_DATO15=='DEATH')
              &((x['Seleziona periodo']=='Feb-2017')
              |(x['Seleziona periodo']=='Mar-2017')
              |(x['Seleziona periodo']=='Apr-2017')),'Value'].sum()
totale = x.loc[(x.Territorio=='Lombardia')&(x.Sesso=='totale') &(x.TIPO_DATO15=='DEATH')&(x['Seleziona periodo']=='2017'), 'Value'].sum()

tables_side(titolo='Data summary of',
            tabs=[df, x],
            tabslabel=['Surface kmq', 'Population and Deaths'])
tabMorti = x[(x.Sesso=='totale') & (x['Seleziona periodo']=='2017')& (x.TIPO_DATO15 == 'DEATH')].loc[:,('Value', 'ITTER107')]
tabPop   = x[(x.Sesso=='totale') & (x['Seleziona periodo']=='2017')& (x.TIPO_DATO15 == 'BEG')].loc[:,('Value', 'ITTER107')]
df = df.merge(tabMorti, left_on='ITTER107' , right_on='ITTER107' , how='left')
df = df.merge(tabPop, left_on='ITTER107' , right_on='ITTER107' , how='left')
df.columns=['Territorio', 'ITTER107', 'Superficie/kmq', 'Morti', 'Popolazione']

df = df.merge(province, left_on='ITTER107' , right_on='ITTER107', how='left').drop(['Territorio_y','Popolazione_y','ExP'], axis=1)
df['DensityPop/kmq'] = round(df.iloc[:,4]/df.iloc[:,2],0)

tables_side(titolo='Data summary for',
            tabs=[df[df.ITTER107.apply(lambda x: len(x))==5],df[df.ITTER107.apply(lambda x: len(x))==4].head(11)],
            tabslabel=['Provinces','Regions'])
dfcause = pd.read_csv('../input/covidcorrelations/CausesDeath2017.csv')
dfcause.loc[(dfcause.CAUSEMORTE_SL.apply(lambda x: '8' in x ))&
            (dfcause.Sesso=='totale')&
            (dfcause.CAUSEMORTE_SL.apply(lambda x: len(x)==3 ))&
            (dfcause.ITTER107.apply(lambda x: len(x)==5))
          ,['ITTER107', 'Territorio',  
            'Causa iniziale di morte - European Short List',  
            'Value']          
           ].sort_values('Value',ascending=False)
dfcause = pd.read_csv('../input/covidcorrelations/CausesDeath2017.csv')
regD = dfcause.loc[(dfcause.CAUSEMORTE_SL.apply(lambda x: '8' in x ))&
            (dfcause.Sesso=='totale')&
            (dfcause.CAUSEMORTE_SL.apply(lambda x: len(x)==3 ))&
            (dfcause.ITTER107.apply(lambda x: len(x)==4))
          ,['ITTER107', 'Territorio',  
            'Causa iniziale di morte - European Short List',  
            'Value']          
           ]
provD = dfcause.loc[(dfcause.CAUSEMORTE_SL.apply(lambda x: '8' in x ))&
            (dfcause.Sesso=='totale')&
            (dfcause.CAUSEMORTE_SL.apply(lambda x: len(x)==3 ))&
            (dfcause.ITTER107.apply(lambda x: len(x)==5))
          ,['ITTER107', 'Territorio',  
            'Causa iniziale di morte - European Short List',  
            'Value']          
           ]
tables_side(titolo='Data summary for',
            tabs=[regD.groupby(['Territorio','Causa iniziale di morte - European Short List']).sum().head(12),
                  provD.groupby(['Territorio','Causa iniziale di morte - European Short List']).sum().head(12),
                  provD.groupby('Causa iniziale di morte - European Short List').sum().sort_values('Value',ascending=False)
            ],
            tabslabel=['Regions','Provinces', 'Total count whole Italy'])
dfcause = dfcause[dfcause['Sesso']=='totale']
dfcause = dfcause[dfcause.CAUSEMORTE_SL.apply(lambda x: len(x)==1)]
listacausemorte = dfcause['Causa iniziale di morte - European Short List'].unique()
codicecausemorte = dfcause[dfcause.CAUSEMORTE_SL.apply(lambda x: len(x)==1)].CAUSEMORTE_SL.unique()

dfx = dfcause.iloc[:,[0,1,5]].groupby(['ITTER107','Territorio','Sesso']).sum()
dfx = dfx.reset_index()

for prov in dfcause.ITTER107.unique():
    for i,ix in zip (listacausemorte, codicecausemorte):
       # print('{}{}'.format(prov,i))
        dfx.loc[dfx.ITTER107==prov,'{} - {}'.format(ix,i)] = dfcause.loc[((dfcause.ITTER107==prov)&(dfcause.Sesso=='totale')&(dfcause.CAUSEMORTE_SL==ix)),'Value'].values[0]
        
df = df.merge(dfx, left_on='ITTER107', right_on='ITTER107', how='left')
df = df.drop(['Territorio', 'Sesso'], axis=1)
df
ap_prov =  pd.read_excel('../input/covidcorrelations/AirPollution2017.xlsx', header=1)

ap_prov[ap_prov['50° percentile1 [µg/m3]'].str.contains('-', na=False)]
ap_prov.drop(ap_prov[ap_prov['50° percentile1 [µg/m3]']=='-'].index, inplace=True)
ap_prov['50° percentile1 [µg/m3]']=pd.to_numeric(ap_prov['50° percentile1 [µg/m3]'])

ap_prov[ap_prov['75° percentile2 [µg/m3]'].str.contains('-', na=False)]
ap_prov.drop(ap_prov[ap_prov['75° percentile2 [µg/m3]']=='-'].index, inplace=True)
ap_prov['75° percentile2 [µg/m3]']=pd.to_numeric(ap_prov['75° percentile2 [µg/m3]'])

dfair = ap_prov.groupby(by='Provincia').max()
dfair.reset_index(inplace=True)
lista_air  =['Aosta', 'Bolzano', 'Forlì -Cesena', 'Massa Carrara','Carbonia-Iglesias']
lista_nuovi_nomi= ["Valle d'Aosta / Vallée d'Aoste",'Bolzano / Bozen','Forlì-Cesena','Massa-Carrara','Sud Sardegna']
for i in np.arange(len(lista_air)):
    dfair = dfair.replace(lista_air[i],lista_nuovi_nomi[i])
    
dfair = dfair.loc[:,('Provincia','Regione','Valore medio annuo1,3 [µg/m³]', '50° percentile1 [µg/m3]','75° percentile2 [µg/m3]')]
dfair


df = df.merge(dfair, left_on='Territorio_x', right_on='Provincia', how='left')
province = df[df.ITTER107.apply(lambda x: len(x)==5)].drop('Provincia', axis=1)
province.reset_index(inplace=True)
province.loc[province.ITTER107=='IT111','MeanAge'] = province[province.Territorio_x=='Cagliari']['MeanAge'].values[0]
province.loc[province.ITTER107=='IT111','PercOver65'] = province[province.Territorio_x=='Cagliari']['PercOver65'].values[0]

province.loc[province.ITTER107.apply(lambda x: 'ITG1' in x),'Regione']= 'Sicilia'
province.loc[(province.ITTER107.apply(lambda x: 'ITG1' in x))&(province['Valore medio annuo1,3 [µg/m³]'].isna()) , 'Valore medio annuo1,3 [µg/m³]'] = 14
province.loc[(province.ITTER107.apply(lambda x: 'ITG1' in x))&(province['50° percentile1 [µg/m3]'].isna()) ,'50° percentile1 [µg/m3]'] = 13
province.loc[(province.ITTER107.apply(lambda x: 'ITG1' in x))&(province['75° percentile2 [µg/m3]'].isna()),'75° percentile2 [µg/m3]'] = 18
province[province.ITTER107.apply(lambda x: 'ITG1' in x)]

province.loc[province.ITTER107.apply(lambda x: 'ITC3' in x),'Regione'] = 'Liguria'
province.loc[(province.ITTER107.apply(lambda x: 'ITC3' in x))&(province['Valore medio annuo1,3 [µg/m³]'].isna()) , 'Valore medio annuo1,3 [µg/m³]'] = 17.3
province.loc[(province.ITTER107.apply(lambda x: 'ITC3' in x))&(province['50° percentile1 [µg/m3]'].isna()) ,'50° percentile1 [µg/m3]'] = 16
province.loc[(province.ITTER107.apply(lambda x: 'ITC3' in x))&(province['75° percentile2 [µg/m3]'].isna()),'75° percentile2 [µg/m3]'] = 21.3
province.loc[province.ITTER107.apply(lambda x: 'ITC3' in x),:]

province.loc[province.ITTER107.apply(lambda x: 'IT10' in x),'Regione'] = 'Lombardia'
province.loc[(province.ITTER107.apply(lambda x: 'IT10' in x))&(province['Valore medio annuo1,3 [µg/m³]'].isna()) , 'Valore medio annuo1,3 [µg/m³]'] = 30
province.loc[(province.ITTER107.apply(lambda x: 'IT10' in x))&(province['50° percentile1 [µg/m3]'].isna()) ,'50° percentile1 [µg/m3]'] = 23
province.loc[(province.ITTER107.apply(lambda x: 'IT10' in x))&(province['75° percentile2 [µg/m3]'].isna()),'75° percentile2 [µg/m3]'] = 39.4
province.loc[province.ITTER107.apply(lambda x: 'IT10' in x),:]

province.loc[province.ITTER107.apply(lambda x: 'ITF5' in x),'Regione'] = 'Basilica'
province.loc[(province.ITTER107.apply(lambda x: 'ITF5' in x))&(province['Valore medio annuo1,3 [µg/m³]'].isna()) , 'Valore medio annuo1,3 [µg/m³]'] = 11
province.loc[(province.ITTER107.apply(lambda x: 'ITF5' in x))&(province['50° percentile1 [µg/m3]'].isna()) ,'50° percentile1 [µg/m3]'] = 10
province.loc[(province.ITTER107.apply(lambda x: 'ITF5' in x))&(province['75° percentile2 [µg/m3]'].isna()),'75° percentile2 [µg/m3]'] = 14
province.loc[province.ITTER107.apply(lambda x: 'ITF5' in x),:]

province.loc[province.ITTER107.apply(lambda x: 'ITF2' in x),'Regione'] = 'Molise'
province.loc[(province.ITTER107.apply(lambda x: 'ITF2' in x))&(province['Valore medio annuo1,3 [µg/m³]'].isna()) , 'Valore medio annuo1,3 [µg/m³]'] = 10
province.loc[(province.ITTER107.apply(lambda x: 'ITF2' in x))&(province['50° percentile1 [µg/m3]'].isna()) ,'50° percentile1 [µg/m3]'] = 10
province.loc[(province.ITTER107.apply(lambda x: 'ITF2' in x))&(province['75° percentile2 [µg/m3]'].isna()),'75° percentile2 [µg/m3]'] = 10
province.loc[province.ITTER107.apply(lambda x: 'ITF2' in x),:]

province
vis = province.sort_values('50° percentile1 [µg/m3]', ascending=False).head(10).loc[:,('Territorio_x', 'Regione', 'Valore medio annuo1,3 [µg/m³]','50° percentile1 [µg/m3]')].groupby('Regione').count()
vis2 = province.sort_values('50° percentile1 [µg/m3]', ascending=False).head(10).loc[:,('Territorio_x', 'Regione', 'Valore medio annuo1,3 [µg/m³]','50° percentile1 [µg/m3]')]
tables_side(titolo='Data summary for',
            tabs=[vis2, vis],
            tabslabel=['The 10 most polluted provinces PM2.5[µg/m3]', 'Count of provinces by Regions'])
for prov in province.ITTER107.unique():
    province.loc[province.ITTER107==prov,'{}/ Popolazione'.format('Valore medio annuo1,3 [µg/m³]')] = province.loc[province.ITTER107==prov,'Valore medio annuo1,3 [µg/m³]'].values[0]*province.loc[province.ITTER107==prov,'Popolazione_x'].values[0]
    province.loc[province.ITTER107==prov,'{}/ Popolazione'.format('50° percentile1 [µg/m3]')] = province.loc[province.ITTER107==prov,'50° percentile1 [µg/m3]'].values[0]*province.loc[province.ITTER107==prov,'Popolazione_x'].values[0]
    province.loc[province.ITTER107==prov,'{}/ Popolazione'.format('75° percentile2 [µg/m3]')] = province.loc[province.ITTER107==prov,'75° percentile2 [µg/m3]'].values[0]*province.loc[province.ITTER107==prov,'Popolazione_x'].values[0]
    for i,ix in zip (listacausemorte, codicecausemorte):
       # print('{}{}'.format(prov,i))
        province.loc[province.ITTER107==prov,'{} - {} %'.format(ix,i)] = province.loc[province.ITTER107==prov,'{} - {}'.format(ix,i)].values[0]/province.loc[province.ITTER107==prov,'Popolazione_x'].values[0]*100


l = ['DensityPop kmq', 'PM25 pollution',
    'circulatory system diseases',
    'some infectious and parasitic diseases',
    'tumors',
    'mental and behavioral disorders',
    'digestive tract diseases',
    'respiratory system diseases',
    'nervous system and sense organs dis.',
    'endocrine,nutritional metabolic dis.',
    'blood,hematopoietic of immune system']

fig, axes = plt.subplots(ncols=3, figsize=(24, 8),dpi=200)
font_scale = 4
ax1,ax2,ax3 = axes

ax1.text(x = 0, y = -2, s = "Correlation between initial cause of death, air pollution and density of population per kmq.", fontsize = 23, weight = 'bold', alpha = .75)
ax1.text(x = 4, y = -0.5, s = "Italy", fontsize = 20, alpha = .75)
ax2.text(x = 4, y = -0.5, s = "Lombardy", fontsize = 20, alpha = .75)
ax3.text(x = 4, y = -0.5, s = "Veneto", fontsize = 20, alpha = .75)
sns.heatmap(
    province.iloc[:,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=l,
    xticklabels=l,
    
    ax=ax1)
sns.heatmap(
    province.iloc[province[province.Regione=='Lombardia'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=False,
    xticklabels=l,
    ax=ax2)
sns.heatmap(
    province.iloc[province[province.Regione=='Veneto'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    yticklabels=False,
    xticklabels=l,  
    linewidth=1,
    cbar=False,
    ax=ax3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
fig, axes = plt.subplots(ncols=3, figsize=(24, 8),dpi=200)
font_scale = 4
ax1,ax2,ax3 = axes

ax1.text(x = 0, y = -2, s = "Correlation between initial cause of death, air pollution and density of population per kmq.", fontsize = 23, weight = 'bold', alpha = .75)
ax1.text(x = 4, y = -0.5, s = "Piemonte", fontsize = 20, alpha = .75)
ax2.text(x = 4, y = -0.5, s = "Lazio", fontsize = 20, alpha = .75)
ax3.text(x = 4, y = -0.5, s = "Sicily", fontsize = 20, alpha = .75)
sns.heatmap(
    province.iloc[province[province.Regione=='Piemonte'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=l,
    xticklabels=l,
    
    ax=ax1)
sns.heatmap(
    province.iloc[province[province.Regione=='Lazio'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=False,
    xticklabels=l,
    ax=ax2)
sns.heatmap(
    province.iloc[province[province.Regione=='Sicilia'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    yticklabels=False,
    xticklabels=l,  
    linewidth=1,
    cbar=False,
    ax=ax3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
fig, axes = plt.subplots(ncols=3, figsize=(24, 8), dpi=200)

ax1,ax2,ax3 = axes

ax1.text(x = 0, y = -2, s = "Correlation between initial cause of death, air pollution and density of population per kmq.", fontsize = 23, weight = 'bold', alpha = .75)
ax1.text(x = 4, y = -0.5, s = "Sardegna", fontsize = 20, alpha = .75)
ax2.text(x = 4, y = -0.5, s = "Marche", fontsize = 20, alpha = .75)
ax3.text(x = 4, y = -0.5, s = "Emilia Romagna", fontsize = 20, alpha = .75)
sns.heatmap(
    province.iloc[province[province.Regione=='Sardegna'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=l,
    xticklabels=l,
    
    ax=ax1)
sns.heatmap(
    province.iloc[province[province.Regione=='Marche'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    linewidth=1,
    cbar=False,
    yticklabels=False,
    xticklabels=l,
    ax=ax2)
sns.heatmap(
    province.iloc[province[province.Regione=='Emilia-Romagna'].index,[8,19,25,26,27,28,29,30,31,32,33]].corr(),
    cmap='coolwarm',
    square=True,
    annot=True,
    cbar_kws={'fraction' : 0.01},
    yticklabels=False,
    xticklabels=l,  
    linewidth=1,
    cbar=False,
    ax=ax3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
tabC=pd.DataFrame()
df_covid = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv')
tabA = df_covid.groupby('denominazione_regione').sum()['deceduti'].reset_index().sort_values('deceduti', ascending=False)
tabA.columns=['Regions', 'Deaths']
tabB = province.groupby(['Regione']).sum().reset_index().iloc[:,[0, 13, 10, 8]].sort_values('8 - malattie del sistema respiratorio', ascending=False)
tabB.columns = ['Regions', 'Respiratory system', 'Tumors', 'Circulatory system']
tabC['Regions'] = tabB['Regions']
tabC['Respiratory system'] = (tabB['Respiratory system']/12*3).apply(lambda x: int(x))
tabC['Tumors'] = (tabB['Tumors']/12*3).apply(lambda x: int(x))
tabC['Circulatory sistem'] = (tabB['Circulatory system']/12*3).apply(lambda x: int(x))

tables_side(titolo='Data summary for Regions in Feb-Mar-Apr --- (in Lombardy Feb/Apr 2017 : {}, in total: {})'.format(trimestre, totale),
            tabs=[tabA, tabC, tabB],
            tabslabel=['Deaths by Coronavirus 2020', 'Deaths in the same period in 2017 for', 'Death in th whole year (2017)'])
import geopandas as gpd
reg_json = gpd.read_file('../input/libraries/reg_ok_2020.geojson')
tabC.loc[tabC.Regions=='Provincia Autonoma Trento','Regions'] = 'P.A. Trento'
tabC.loc[tabC.Regions=='Provincia Autonoma Bolzano','Regions'] = 'P.A. Bolzano'
tabC.loc[tabC.Regions=='Friuli Venezia-Giulia','Regions'] = 'Friuli Venezia Giulia'
tabC.loc[tabC.Regions=='Basilica','Regions'] = 'Basilicata'
reg_json.loc[reg_json.Regione=='Emilia Romagna','Regione'] = 'Emilia-Romagna'
vis = reg_json.merge(tabA, how='left', left_on='Regione', right_on = 'Regions')
vis2 = reg_json.merge(tabC, how='left', left_on='Regione', right_on = 'Regions')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

ax[0].set_title('Covid Deaths')
vis.plot(column='Deaths', ax=ax[0], cmap='magma',scheme='BoxPlot')

ax[1].set_title('Respiratory Diseases Deaths')
vis2.plot(column='Respiratory system', ax=ax[1], cmap='magma',scheme='BoxPlot')

plt.show()
from sklearn.cluster import KMeans
model = KMeans(n_clusters= 5)
p = province.loc[:, ['Territorio_x', 'Regione', 'Superficie/kmq', 'MeanAge', 'DensityPop/kmq','50° percentile1 [µg/m3]',
       'Valore medio annuo1,3 [µg/m³]/ Popolazione',
       '7 - malattie del sistema circolatorio %',
       '1 - alcune malattie infettive e parassitarie %', '2 - tumori %',
       '5 - disturbi psichici e comportamentali %',
       "9 - malattie dell'apparato digerente %",
       '8 - malattie del sistema respiratorio %',
       '6 - malattie del sistema nervoso e degli organi di senso %',
       '4 - malattie endocrine, nutrizionali e metaboliche %',
       '3 - malattie del sangue e degli organi ematopoietici ed alcuni disturbi del sistema immunitario %']]
model.fit(p.iloc[:,2:])
p['lab']= model.labels_
fig = plt.figure()
g = sns.pairplot(data=p.loc[:,['DensityPop/kmq',
       '7 - malattie del sistema circolatorio %',
       '8 - malattie del sistema respiratorio %','lab']], hue='lab', height=5,)
