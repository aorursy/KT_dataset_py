from IPython.display import Image

Image(filename='../input/images/PM25-COVID19.png')
import pandas as pd
import warnings
from IPython.display import HTML

def tables_side(titolo, tabs, tabslabel):
    return HTML( '<table><caption style="background-color:#F1F2F3;"><h1>'+titolo+'</h1></caption>'+
                 '<tr style="background-color:#D9D9D9;" >'+''.join(['<td><h2 align="center">'+tlabel+ '</h2></td>' for tlabel in tabslabel]) +'</tr>'+
                 '<tr style="background-color:#D9D9D9;">'+''.join(['<td>'+table._repr_html_() + '</td>' for table in tabs]) +
                 '</tr></table>'
               )
warnings.filterwarnings('ignore')

df_demo = pd.read_csv('../input/dataset/Regions.csv', index_col=0)
df_covid = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-latest.csv')
df_air = pd.read_excel('../input/airpollution/Tabella 3.xlsx', header=1)
df_demo.head()
df_covid.loc[df_covid.denominazione_regione=='Emilia-Romagna','denominazione_regione'] = 'Emilia Romagna'
df_air
df_covid.groupby('denominazione_regione').sum()
# Uniform the names of the regions in the datasets
df_air[df_air['max media annuale (mg/m3)'].str.contains('n', na=False)]
df_air.drop(df_air[df_air['max media annuale (mg/m3)']=='n.d.'].index, inplace=True)
df_air[df_air['Regione']=='SICILIA']
df_air['Regione'] = df_air.iloc[:,2].str.replace('_', ' ')
df_air['Regione'] = df_air.iloc[:,2].str.replace('PA ', 'P.A. ')
df_air['Regione'] = df_air.iloc[:,2].apply(str.title)
df_air2 = pd.read_excel('../input/airpollution/Tabella 2.xlsx', header=1)
df_basilicata = df_air2[df_air2['Regione']=='Basilicata']
data_basilicata=['Not Important',17, 'Basilicata', 'Rurale', 'Not Important', 'No',10]
data_basilicata = pd.DataFrame([pd.Series(data_basilicata)])
data_basilicata.columns = df_air.columns
df_air = pd.concat([df_air, data_basilicata], ignore_index=True)
df_air.loc[df_air['Regione']=='Valle Aosta', 'Regione']= "Valle d'Aosta"
df_air.iloc[:,6]= pd.to_numeric(df_air.iloc[:,6])
df_air
df_summary = df_covid[['codice_regione', 'denominazione_regione', 'deceduti']]
df_summary = df_summary.merge(df_demo, left_on='denominazione_regione', right_on='denominazione_regione')
df_summary['death%'] = df_summary['deceduti']/df_summary['popolazione']*100
df_air_med = df_air.groupby(['Regione']).agg({'max media annuale (mg/m3)': ['count', 'mean', 'median']})
tables_side(titolo='Comparison between',
            tabs=[df_summary.sort_values(by='death%', ascending=False), df_air_med.sort_values(by=('max media annuale (mg/m3)', 'median'), ascending=False)],
            tabslabel=['Summary COVID ord by Death%','Summary Air Pollution'])
df_summary = df_summary.sort_values(by='death%', ascending=False).merge(df_air_med, left_on='denominazione_regione',right_on='Regione')
df_summary = df_summary.iloc[:,[0,1,2,3,4,7]]
df_summary.columns = ['codice_regione',
               'denominazione_regione',
                            'deceduti',
                         'popolazione',
                              'death%',
                     'air_poll_median']
df_summary
df_covid_complete = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
df_covid_complete.head(3)

df_covid_complete.loc[df_covid_complete.denominazione_regione=='Emilia-Romagna','denominazione_regione'] = 'Emilia Romagna'
list_Regions = df_demo.denominazione_regione
dfCVprop = df_covid_complete[['data', 'denominazione_regione', 'deceduti','totale_ospedalizzati']]
for i in list_Regions:
    dfCVprop.loc[dfCVprop.denominazione_regione==i,'deceduti'] = dfCVprop.loc[dfCVprop.denominazione_regione==i,'deceduti']/df_demo[df_demo.denominazione_regione==i]['popolazione'].values*100
    dfCVprop.loc[dfCVprop.denominazione_regione==i,'totale_ospedalizzati'] = dfCVprop.loc[dfCVprop.denominazione_regione==i,'totale_ospedalizzati']/df_demo[df_demo.denominazione_regione==i]['popolazione'].values*100
df_main = dfCVprop.pivot_table(index='denominazione_regione', columns='data', values='deceduti').transpose()
df=df_covid_complete
lista_Regioni = df.denominazione_regione.unique()
tot = len(df[df['denominazione_regione']=='Lombardia'].loc[:,'deceduti'])
for l in lista_Regioni:
    dec = df[df.denominazione_regione==l]['deceduti'].index
    val =df[df.denominazione_regione==l]['deceduti'].values
    valosp =df[df.denominazione_regione==l]['totale_ospedalizzati'].values
    a = val[0]
    b = valosp[0]
    for x in range(1,tot):
        df.loc[dec[x-1],'nuovi_dec'] = a
        df.loc[dec[x-1],'nuovi_osp'] = b
        a = val[x]-val[x-1]
        b = valosp[x]-valosp[x-1]
    df.loc[dec[tot-1],'nuovi_dec'] = val[tot-1]-val[tot-2]
    df.loc[dec[tot-1],'nuovi_osp'] = valosp[tot-1]-valosp[tot-2]

df_summary
import matplotlib.pyplot as plt
import seaborn as sns

l = ['Lombardia', 'Emilia Romagna', 'Marche', "Valle d'Aosta", 'Toscana', 'Campania']
datagiorn=df.data.str[5:10]
plt.figure(figsize=(20,10))
for i, r in zip (l, range(len(l))):
    plt.subplot(2,3,r+1)
    sns.lineplot(x=datagiorn.unique(), y=df[df.denominazione_regione==i]['nuovi_dec'], label='Progression of deaths')
    sns.lineplot(x=datagiorn.unique(), y=df[df.denominazione_regione==i]['nuovi_osp'], label='Hospitalized')
    plt.title(i)
    plt.xticks(rotation=90)
fig = plt.figure(figsize=(24,15))
ax1 = fig.add_subplot(121)

for i  in list_Regions:
    ax1=sns.lineplot(x=df_main.index, y=df_main.loc[:,i], label=i)
#ax1.axvline('2020-03-07T18:00:00', 0, 1, label='pyplot vertical line', ls='--')
ax1.grid(color='gray', linestyle='-', linewidth=0.3)
ax1.annotate('start of cases', xy=('2020-03-07T18:00:00', 0.001),
             xycoords='data',
             size = 15,
             xytext=(0, 0.01),
             textcoords='data',
             arrowprops=dict(arrowstyle= '-|>',
                             color='r',
                             lw=5,
                             ls='-')
           )
ax1.legend()

plt.xticks(rotation=90)
plt.title('Death % Pop Progression')

ax2 = fig.add_subplot(122)
font_size=14
bbox=[0, 0, 1, 1]
ax2.axis('off')
plt.title('Table Max Death % Pop per Regions')

mpl_table = ax2.table(cellText = df_summary.iloc[:,[1,4]].sort_values(by='death%',ascending=False).values, rowLabels = df_summary.iloc[:,[1,4]].index, bbox=bbox, colLabels=['Regione','Morti in % Popolazione'],)
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(font_size)
heatmapshow = df_summary[['denominazione_regione',
                                       'deceduti',
                                         'death%',
                                    'popolazione',
                                'air_poll_median']].corr()
plt.figure(figsize=(20,10))
plt.subplot(122)
sns.scatterplot(x=df_summary.iloc[:,5], y= df_summary['death%'], data=df_summary, hue='air_poll_median', s=500)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(121)
sns.heatmap(heatmapshow, annot=True, annot_kws={'size':20})
pivtable = df_covid_complete.pivot_table(index='denominazione_regione', columns='data', values='deceduti')
sns.clustermap(pivtable, cmap='coolwarm', standard_scale=1)
plt.title('Clustermap Normalizzato')

sns.clustermap(pivtable, cmap='coolwarm')
plt.title('Clutermap NON normalizzato')
plt.show()
import matplotlib.gridspec
i='Emilia Romagna'
sns.clustermap(df_main.transpose(), cmap='coolwarm', standard_scale=1, figsize=(10,10))
plt.title('Clustermap Normalizzato')
l = ['Lombardia', 'Emilia Romagna', 'Marche', "Valle d'Aosta", 'Toscana', 'Campania', 'Molise']
g = sns.clustermap(df_main.transpose(), cmap='coolwarm',figsize=(20,10))
g.gs.update(left=0.05, right=0.45)
plt.title('Clutermap NON normalizzato')
gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.6)
ax2 = g.fig.add_subplot(gs2[0])
for i in l:
    sns.lineplot(x=df_main.index, y=df_main.loc[:,i], label=i, ax=ax2)
plt.ylabel('Death % population')
plt.grid(color='gray', linestyle='-', linewidth=0.3)
plt.xticks(rotation=90)
plt.legend(loc='upper left')


plt.show()
import geopandas as gpd
# the json libraries that I found on the net do not have the same names of provinces and regions of my tables
# I uniformed to be able to relate them not to lose data.
# I want to display graphs both by region and by province

# the original file can be found here, https://gist.github.com/datajournalism-it/f1abb68e718b54f6a0fe ,
# my versions ARE UNIFORMED with my tables

reg_json = gpd.read_file('../input/libraries/reg_ok_2020.geojson')

# the original file for the provinces can be found here https://gist.github.com/datajournalism-it/212e7134625fbee6f9f7
prov_json = gpd.read_file('../input/libraries/prov_ok_2020.geojson')
vis = reg_json.merge(df_summary, how='left', left_on='Regione', right_on = 'denominazione_regione')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

ax[0][0].set_title('Air Pollution PM2.5')
vis.plot(column='air_poll_median', ax=ax[0][0], cmap='coolwarm', legend=True)
ax[0][1].set_title('Death % population')
vis.plot(column='death%', ax=ax[0][1], cmap='coolwarm', legend=True)
ax[1][0].set_title('Air Pollution PM2.5 - BoxPlot Legend')
vis.plot(column='air_poll_median', ax=ax[1][0], cmap='coolwarm', legend=True, scheme='BoxPlot')
ax[1][1].set_title('Death % population - BoxPlot Legend')
vis.plot(column='death%', ax=ax[1][1], cmap='coolwarm', legend=True, scheme='BoxPlot')

plt.show()

df_summary
ap_prov =  pd.read_excel('../input/airpollution/Tabella 2.xlsx', header=1)

ap_prov[ap_prov['50° percentile1 [µg/m3]'].str.contains('-', na=False)]
ap_prov.drop(ap_prov[ap_prov['50° percentile1 [µg/m3]']=='-'].index, inplace=True)
ap_prov['50° percentile1 [µg/m3]']=pd.to_numeric(ap_prov['50° percentile1 [µg/m3]'])

ap_prov[ap_prov['75° percentile2 [µg/m3]'].str.contains('-', na=False)]
ap_prov.drop(ap_prov[ap_prov['75° percentile2 [µg/m3]']=='-'].index, inplace=True)
ap_prov['75° percentile2 [µg/m3]']=pd.to_numeric(ap_prov['75° percentile2 [µg/m3]'])

df = ap_prov.groupby(by='Provincia').median()
df.columns = ['Region_id', 'Province_id', 'Municipality_id', 'Observation_id',
       'Station_code', 'airpoll_50perc', 'airpoll_75perc', 'Numero di dati validi']
df.reset_index(inplace=True)
df.head(5)
df_covid_prov = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv')
df_covid_prov.loc[df_covid_prov['denominazione_provincia']=='Forlì-Cesena','denominazione_provincia'] = 'Forlì -Cesena'
df_covid_prov.loc[df_covid_prov['denominazione_provincia']=='In fase di definizione/aggiornamento','denominazione_provincia']
df_covid_prov.drop(df_covid_prov[df_covid_prov['denominazione_provincia']=='In fase di definizione/aggiornamento'].index, inplace=True)
df_prov = df_covid_prov.groupby('denominazione_provincia').median()
df_prov.reset_index(inplace=True)
df = df_prov.merge(df, how='left', left_on='denominazione_provincia', right_on='Provincia')
df.head(3)
df = df.loc[:,['denominazione_provincia','totale_casi','airpoll_50perc', 'airpoll_75perc']]
df = df.loc[:,['denominazione_provincia','totale_casi','airpoll_50perc', 'airpoll_75perc']]
df['airpoll_50perc'].dropna(inplace=True)
df['airpoll_75perc'].dropna(inplace=True)
vis_prov = prov_json.merge(df, how='left', left_on='NOME_PRO', right_on = 'denominazione_provincia')
vis_prov.dropna(inplace=True)
a = vis_prov.sort_values('totale_casi', ascending=False).head(10).loc[:,['denominazione_provincia','totale_casi']]
b = vis_prov.sort_values('airpoll_50perc', ascending=False).head(10).loc[:,['denominazione_provincia','airpoll_50perc']]
c = vis_prov.sort_values('airpoll_75perc', ascending=False).head(10).loc[:,['denominazione_provincia','airpoll_75perc']]
tables_side(titolo='Confronto provincie', tabs=[a,b,c], tabslabel=['Total cases','Pollution PM 2.5 - 50 percentile','Pollution PM 2.5 - 75 percentile'])
tableAP =pd.concat([b.reset_index(drop=True), c.reset_index(drop=True)], axis=1)
tableAP.loc[:, 'airpoll_50perc'] = tableAP.loc[:, 'airpoll_50perc'].round(2)
tableAP.loc[:, 'airpoll_75perc'] = tableAP.loc[:, 'airpoll_75perc'].round(2)
tableAP.columns =['prov1', 'airpoll_50perc', 'prov2', 'airpoll_75perc']
tableAP.loc[tableAP.prov1=='Monza e della Brianza','prov1'] = 'Monza Brianza'
tableAP.loc[tableAP.prov2=='Monza e della Brianza','prov2'] = 'Monza Brianza'
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,16))
ax[0][0].set_title('Air Pollution PM2.5 - 50 percentile')
vis_prov.plot(column='airpoll_50perc', ax=ax[0][0], cmap='coolwarm', legend=True )
ax[0][1].set_title('Total COVID cases')
vis_prov.plot(column='totale_casi', ax=ax[0][1], cmap='coolwarm', legend=True)
ax[1][0].set_title('Air Pollution PM2.5 - 75 percentile')
vis_prov.plot(column='airpoll_75perc', ax=ax[1][0], cmap='coolwarm', legend=True)

ax[1][1].axis('off')
table = ax[1][1].table(cellText = tableAP.values, rowLabels = tableAP.index, bbox=bbox, colLabels=['Province','Air Poll 50 percentile', 'Province', 'Air Poll 75 percentile'],)
table.auto_set_font_size(False)
mpl_table.set_fontsize(14)

plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,16))

ax[0][0].set_title('Air Pollution PM2.5 - 50 percentile')
vis_prov.plot(column='airpoll_50perc', ax=ax[0][0], cmap='coolwarm', legend=True , scheme='box_plot')
ax[0][1].set_title('Total COVID cases')
vis_prov.plot(column='totale_casi', ax=ax[0][1], cmap='coolwarm', legend=True, scheme='box_plot')
ax[1][0].set_title('Air Pollution PM2.5 - 75 percentile')
vis_prov.plot(column='airpoll_75perc', ax=ax[1][0], cmap='coolwarm', legend=True , scheme='box_plot')

ax[1][1].axis('off')
table = ax[1][1].table(cellText = tableAP.values, rowLabels = tableAP.index, bbox=bbox, colLabels=['Province','Air Poll 50 percentile', 'Province', 'Air Poll 75 percentile'],)
table.auto_set_font_size(False)
mpl_table.set_fontsize(14)

plt.show()
!pip install lmfit
# libraries to manage number
import numpy as np
import pandas as pd
# libraries to plotting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# libraries to model

# Libraries to metrics
from sklearn import metrics


# libraries to optimize model
from lmfit import Model
def sigmoid(x, b, r, t):
    z = (t * (b + x))
    sig = 1/(1 + np.exp( -z ))*r
    return sig
#http://www.edscave.com/forecasting---time-series-metrics.html
def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
df.head()
days = df.loc[df.denominazione_regione=='Lombardia','data'].shape
days = np.arange(days[0])
region_list = df.denominazione_regione.unique()
region_list
fig, ax = plt.subplots(nrows=21, ncols=2, figsize=(16,80))
n = 0
df_coeff = pd.DataFrame()
for reg in region_list:
    ax[n][0].set_title('Current situation {}'.format(reg))
    sns.scatterplot(x=days, y=df.loc[df.denominazione_regione==reg,'deceduti'].values,ax=ax[n][0])
    x = days
    y = df.loc[df.denominazione_regione==reg,'deceduti'].values
           
    model = Model(sigmoid)
    pred = model.fit(y, x=x, b= 0, r=y.max()/2, t= 0.001)

    x_ideal = np.linspace(np.min(x), np.max(x)*2)
    ideal = pred.eval(x=x_ideal)
    sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[n][1])
    

    sns.scatterplot(x=x, y=y, label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[n][1])

    ax[n][1].grid()
    ax[n][1].set_title('{} model'.format(reg))
    ax[n][1].set_xlim(0)
    
    df_coeff = df_coeff.append(pred.values, ignore_index=True)
    #df_coeff.loc[n, 'MAPE'] = mean_absolute_percentage_error(ideal[:len(y)],y)
    df_coeff.loc[n,'Regions'] = reg
    n+=1

#plt.savefig('plot_fig.png')   
#plt.show()

italy = df.groupby('data').sum()

reg='Italy'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].set_title('Current situation {}'.format(reg))
sns.scatterplot(x=days, y=italy['deceduti'].values,ax=ax[0])
x = days
y = italy['deceduti'].values
        
model = Model(sigmoid)
pred = model.fit(y, x=x, b= 0, r=y.max()/2, t= 0.001)
x_ideal = np.linspace(np.min(x), np.max(x)*2)
ideal = pred.eval(x=x_ideal)
sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[1])
sns.scatterplot(x=x, y=y, label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[1])

ax[1].grid()
ax[1].set_title('{} model'.format(reg))
ax[1].set_xlim(0)

df_coeff = df_coeff.append(pred.values, ignore_index=True)
#df_coeff.loc[n, 'MAPE'] = mean_absolute_percentage_error(ideal[:len(y)],y)
df_coeff.loc[21,'Regions'] = reg
df_coeff.columns=['beta', 'cap', 'theta', 'Regions']
df_coeff['cap'] = df_coeff['cap'].apply(lambda x: int(x))
df_coeff[['cap','Regions']].rename(columns={'cap':'Assumed number of deaths','Regions':'Regions'})
print('Forecasting with regions study: {}'.format(df_coeff.cap[:21].sum()))
print('Forecasting with regions study: {}'.format(df_coeff.cap[21]))
from scipy.spatial import ConvexHull
def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
df_coeff
plt.figure(figsize=(10,10))
sns.scatterplot(data=df_coeff, x='beta', y='theta', s=100)
n = list(np.concatenate( (df_coeff.sort_values('theta', ascending=False).head(3).index.values,df_coeff.sort_values('beta').head(2).index.values), axis=0))
for i in n:
    plt.annotate(df_coeff.Regions[i], (df_coeff.beta[i], df_coeff.theta[i]*1.01),size=20)

N = list(df_coeff.index.values)
N = [x for x in N if x not in n]
sns.scatterplot(data=df_coeff[df_coeff.index==21], x='beta', y='theta', s=200, color='red')
plt.annotate(df_coeff.Regions[21], (df_coeff.beta[21], df_coeff.theta[21]*1.01),size=20)
encircle(df_coeff[df_coeff.index.isin(N)].beta, df_coeff[df_coeff.index.isin(N)].theta, ec="orange", fc="none")
model = Model(sigmoid)
pred = model.fit([1,2,3,4], x=[1,2,3,4], b= 0, r=y.max()/2, t= 0.001)
