import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display

pd.options.display.max_columns = None

pd.options.display.max_rows = None

import seaborn as sns

import folium
df = pd.read_csv('../input/sudeste.csv')
#df.to_hdf('weather.hdf','mydata',mode='w')
#df = pd.read_hdf('weather.hdf','mydata')
df.info()
df.describe()
df.shape
df = df.set_index('mdct')

df.index = pd.to_datetime(df.index)
df.isna().sum()
is_178 = df['wsid'] == 178

w_station_178 = df[is_178]
w_station_178.plot(x='date', y='temp')
is_384 = df['wsid'] == 384

station_384 = df[is_384]

station_384_tail_365 = station_384.tail(365)

station_384_tail_365 = station_384_tail_365.reset_index()

station_384_tail_365.plot(x='mdct', y='temp')

station_384_tail_365.plot(x='mdct', y='prcp')

station_384_tail_365.plot(x='mdct', y='hmdy')

station_384_tail_365.plot(x='mdct', y='gbrd')

plt.rcParams['figure.figsize'] = (20, 10)

station_384_tail_365[['wdsp', 'gust', 'temp', 'tmax']].plot()

station_384_tail_365[['temp', 'hmdy']].plot()
station_384_tail_365[['temp']].plot()
station_384_temp = station_384_tail_365['temp']

station_384_temp.replace(0, np.nan, inplace=True)
station_384_temp.plot()
station_384_temp.interpolate('linear', inplace=True, limit_direction='both')
station_384_temp.plot()
station_384_temp_sept_oct = station_384_temp.head(24)

station_384_temp_sept_oct.plot()
is_385 = df['wsid'] == 385

station_385 = df[is_385]

station_385_year = station_385.tail(8760)
pd.options.mode.chained_assignment = None  # default='warn'

station_385_year_temp = station_385_year['temp']

station_385_year_temp.replace(0, np.nan, inplace=True)

station_385_year_temp.interpolate('linear', inplace=True, limit_direction='both')

plt.rcParams['figure.figsize'] = (40, 20)

station_385_year_temp.plot()
station_385_year[['gbrd']].plot()
rain_df = station_385_year[['mo','prcp']].fillna(0)

total_rain = rain_df.groupby('mo').sum().reset_index()

average_rain = rain_df.groupby('mo').mean().reset_index()

average_rain.plot( x='mo', y='prcp', kind='bar')
sns.barplot(x=total_rain.mo, y=total_rain.prcp)
sns.barplot(x=average_rain.mo, y=average_rain.prcp)
is_385 = df['wsid'] == 385

station_385 = df[is_385]
station_385.shape
station_385['temp'].replace(0, np.nan, inplace=True)

station_385['temp'].interpolate('linear', inplace=True, limit_direction='both')

station_385['prcp'].fillna(0,inplace=True)

station_385['gbrd'].fillna(0,inplace=True)

station_385['tmax'].replace(0, np.nan, inplace=True)

station_385['tmax'].interpolate('linear', inplace=True, limit_direction='both')

station_385['tmin'].replace(0, np.nan, inplace=True)

station_385['tmin'].interpolate('linear', inplace=True, limit_direction='both')

station_385['dmax'].replace(0, np.nan, inplace=True)

station_385['dmax'].interpolate('linear', inplace=True, limit_direction='both')

station_385['dewp'].replace(0, np.nan, inplace=True)

station_385['dewp'].interpolate('linear', inplace=True, limit_direction='both')

station_385['dmin'].replace(0, np.nan, inplace=True)

station_385['dmin'].interpolate('linear', inplace=True, limit_direction='both')

station_385['hmax'].replace(0, np.nan, inplace=True)

station_385['hmax'].interpolate('linear', inplace=True, limit_direction='both')

station_385['hmin'].replace(0, np.nan, inplace=True)

station_385['hmin'].interpolate('linear', inplace=True, limit_direction='both')
d = {}

for col in station_385:

    d[col] = station_385[col].unique().tolist()

    

print(d['prcp'])
station_385.isna().sum()
col = ['prcp', 'stp', 'smax', 'smin', 'gbrd', 'temp',

       'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp',

       'wdct', 'gust']

station_385 = station_385[(station_385[col] != 0).any(axis=1)]
station_385.to_hdf('station_385.hdf','mydata',mode='w')
df = pd.read_csv('../input/sudeste.csv')

#df = pd.read_hdf('weather.hdf','mydata')

df = df.set_index('mdct')

df.index = pd.to_datetime(df.index)

df.isna().sum()
(df == 0).astype(int).sum(axis=0)
df['prcp'].fillna(0,inplace=True)

df['gbrd'].fillna(0,inplace=True)
col = ['prcp', 'stp', 'smax', 'smin', 'gbrd', 'temp',

       'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp',

       'wdct', 'gust']

df = df[(df[col] != 0).any(axis=1)]
df.shape
df['temp'].replace(0, np.nan, inplace=True)

df['temp'].interpolate('linear', inplace=True, limit_direction='both')

df['dewp'].replace(0, np.nan, inplace=True)

df['dewp'].interpolate('linear', inplace=True, limit_direction='both')

df['tmax'].replace(0, np.nan, inplace=True)

df['tmax'].interpolate('linear', inplace=True, limit_direction='both')

df['tmin'].replace(0, np.nan, inplace=True)

df['tmin'].interpolate('linear', inplace=True, limit_direction='both')

df['dmax'].replace(0, np.nan, inplace=True)

df['dmax'].interpolate('linear', inplace=True, limit_direction='both')

df['dmin'].replace(0, np.nan, inplace=True)

df['dmin'].interpolate('linear', inplace=True, limit_direction='both')

df['hmax'].replace(0, np.nan, inplace=True)

df['hmax'].interpolate('linear', inplace=True, limit_direction='both')

df['hmin'].replace(0, np.nan, inplace=True)

df['hmin'].interpolate('linear', inplace=True, limit_direction='both')

df['hmdy'].replace(0, np.nan, inplace=True)

df['hmdy'].interpolate('linear', inplace=True, limit_direction='both')

df['wdsp'].interpolate('linear', inplace=True,limit_direction='both')

df['gust'].interpolate('linear', inplace=True,limit_direction='both')
plt.rcParams['figure.figsize'] = (20, 10)

is_384 = df['wsid'] == 384

station_384 = df[is_384]

station_384_year = station_384.tail(365)



station_384_year[['wdsp', 'gust', 'temp', 'tmax']].plot()

station_384_year[['temp', 'hmdy']].plot()
df.isna().sum()
(df == 0).astype(int).sum(axis=0)
df[df.elvt == 0].index[0]
df[df.elvt == 0].head()
df[["wsid", "wsnm", "city", "prov", "elvt", "lat", "lon"]].drop_duplicates()
df[df.elvt == 0].wsid.unique()
df['elvt'].replace(0, 4, inplace=True)

df['lat'].replace(0, -23.993611, inplace=True)

df['lon'].replace(0, -46.256389, inplace=True)
stp_smax_equal = df[df['stp'] == df['smax']].count().stp

stp_smax_equal
stp_smin_equal = df[df['stp'] == df['smin']].count().stp

stp_smin_equal
smax_smin_equal = df[df['smax'] == df['smin']].count().stp

smax_smin_equal
df['stp'].replace(0, np.nan, inplace=True)

df['stp'].interpolate('linear', inplace=True, limit_direction='both')

df['smax'].replace(0, np.nan, inplace=True)

df['smax'].interpolate('linear', inplace=True, limit_direction='both')

df['smin'].replace(0, np.nan, inplace=True)

df['smin'].interpolate('linear', inplace=True, limit_direction='both')
is_384 = df['wsid'] == 384

station_384 = df[is_384]

station_384_tail_365 = station_384.tail(365)

station_384_tail_365 = station_384_tail_365.reset_index()





station_384_tail_365.plot(x='mdct', y='wdsp')

#station_384_tail_365.plot(x='mdct', y='wdct')

station_384_tail_365.plot(x='mdct', y='gust')

station_384_tail_365.plot(x='mdct', y='stp')

station_384_tail_365.plot(x='mdct', y='smax')

station_384_tail_365.plot(x='mdct', y='smin')

is_384 = df['wsid'] == 384

station_384 = df[is_384]

station_384_tail_365 = station_384.tail(24)

station_384_tail_365 = station_384_tail_365.reset_index()



station_384_tail_365.plot(x='mdct', y='wdsp')

#station_384_tail_365.plot(x='mdct', y='wdct')

station_384_tail_365.plot(x='mdct', y='gust')

station_384_tail_365.plot(x='mdct', y='stp')

station_384_tail_365.plot(x='mdct', y='smax')

station_384_tail_365.plot(x='mdct', y='smin')
a = df[['wsnm','lat','lon']].drop_duplicates()

for i in range(len(a)):

    print("folium.Marker([{}, {}], popup='<b>{}</b>', tooltip=tooltip).add_to(m)".format(a.lat.values[i], a.lon.values[i], a.wsnm.values[i]))

m = folium.Map(

    location=[-19.481935, -45.593932],

    zoom_start=4,

    tiles='Stamen Terrain'

)



tooltip = 'Click me!'



folium.Marker([-6.835777, -38.311583], popup='<b>SÃO GONÇALO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.271094, -40.306069], popup='<b>VITÓRIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.988388, -40.579572], popup='<b>SANTA TERESA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.356923000000002, -40.06868], popup='<b>LINHARES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.636526, -40.741818], popup='<b>ALFREDO CHAVES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.676198, -39.864051], popup='<b>SÃO MATEUS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.750412, -41.488852], popup='<b>ALEGRE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.100804999999998, -41.039395], popup='<b>PRES. KENNEDY</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.695265, -40.390572], popup='<b>NOVA VENECIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.104194, -41.106861], popup='<b>AFONSO CLAUDIO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.228373, -43.767703000000004], popup='<b>BARBACENA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.605696, -46.949616999999996], popup='<b>ARAXÁ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.686316, -43.843763], popup='<b>MONTES CLAROS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.917072, -48.255657], popup='<b>UBERLÂNDIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.166773000000003, -40.687756], popup='<b>ALMENARA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.861604, -46.043365], popup='<b>MONTE VERDE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.762607, -42.864013], popup='<b>VIÇOSA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.573824, -42.622413], popup='<b>TIMÓTEO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.952907, -49.525071999999994], popup='<b>ITUIUTABA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.556592000000002, -43.75622], popup='<b>OURO BRANCO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.106502, -44.250928], popup='<b>S.J.DEL REI</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.566513, -45.4043], popup='<b>VARGINHA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.745237, -46.633916], popup='<b>PASSOS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.104867000000002, -42.375904], popup='<b>MURIAÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.769965, -43.364329], popup='<b>JUÍZ DE FORA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.53921, -49.518133], popup='<b>CAMPINA VERDE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.98586, -48.151574], popup='<b>CONC. DAS ALAGOAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.883945, -43.969397], popup='<b>BELO HOR. (PAMPULHA)</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.798799, -40.249933], popup='<b>SERRA DOS AIMORES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.996684, -46.985935], popup='<b>PATROCÍNIO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.454929999999997, -45.453825], popup='<b>FORMIGA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.875270999999998, -47.434102], popup='<b>SACRAMENTO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-14.408228, -44.404123], popup='<b>MONTALVANIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.89284, -41.515423], popup='<b>TEÓFILO OTONI</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.200854999999997, -45.459835999999996], popup='<b>TRÊS MARIAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.395799, -44.961909000000006], popup='<b>PASSA QUATRO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.918066, -46.382996], popup='<b>CALDAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.314567, -45.373049], popup='<b>MARIA DA FÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.830354, -41.977007], popup='<b>GOVERNADOR VALADARES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.786842, -42.942921000000005], popup='<b>GUANHÃES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.532768, -41.090801], popup='<b>AIMORES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.885398000000002, -44.416883], popup='<b>FLORESTAL</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.481935, -45.593932], popup='<b>DORES DO INDAIÁ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.231052, -43.648269], popup='<b>DIAMANTINA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.747711, -44.453784999999996], popup='<b>CURVELO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.085957999999998, -44.016036], popup='<b>MOCAMBINHO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-18.780620000000003, -40.986505], popup='<b>MANTENA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.705504, -42.389256], popup='<b>CAPELINHA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.554101, -46.881935], popup='<b>UNAI</b>', tooltip=tooltip).add_to(m)

folium.Marker([-14.912284, -42.808473], popup='<b>ESPINOSA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.524262, -46.435536], popup='<b>BURITIS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.258064, -44.8356], popup='<b>PIRAPORA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.561349, -47.199251000000004], popup='<b>GUARDA-MOR</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.362747, -45.123801], popup='<b>SÃO ROMÃO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.300158999999999, -45.617430999999996], popup='<b>CHAPADA GAÚCHA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.751535999999998, -41.457786999999996], popup='<b>ÁGUAS VERMELHAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.575667000000003, -41.485588], popup='<b>ITAOBIM</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.723139000000002, -42.43575], popup='<b>RIO PARDO DE MINAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-16.160327, -42.310266], popup='<b>SALINAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-17.78477, -46.119394], popup='<b>JOÃO PINHEIRO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.735757, -42.137146], popup='<b>CARATINGA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.031457, -44.011249], popup='<b>IBIRITÈ (ROLA MOÇA)</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.263378, -42.182817], popup='<b>MANHUAÇU</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.546728, -43.261029], popup='<b>CORONEL PACHECO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-15.448089000000001, -44.366313], popup='<b>JANUARIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.239555, -45.008187], popup='<b>POMPEU</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.909892000000003, -47.114259000000004], popup='<b>SÃO SEBASTIAO DO PARAÍSO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.455288, -44.17338], popup='<b>SETE LAGOAS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-19.980034, -43.958651], popup='<b>BELO HORIZONTE - CERCADINHO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.757868, -43.684843], popup='<b>ECOLOGIA AGRÍCOLA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.050334, -43.595684999999996], popup='<b>MARAMBAIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.589806, -43.282233000000005], popup='<b>XEREM</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.587729, -41.958327000000004], popup='<b>CAMBUCÍ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.975468, -42.02145], popup='<b>ARRAIAL DO CABO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.041647, -41.051871000000006], popup='<b>CAMPOS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.376317999999998, -41.812053000000006], popup='<b>MACAÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.450924, -44.444786], popup='<b>RESENDE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.464889000000003, -43.291505], popup='<b>PICO DO COUTO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.358094, -43.695656], popup='<b>VALENÇA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.448922, -42.987146], popup='<b>TERESÓPOLIS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.22354, -44.726863], popup='<b>PARATÍ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.041647, -41.051871000000006], popup='<b>SÃO TOMÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.861322, -43.41141], popup='<b>VILA MILITAR</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.334839000000002, -42.676932], popup='<b>NOVA FRIBURGO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.098391, -43.208574], popup='<b>TRÊS RIOS </b>', tooltip=tooltip).add_to(m)

folium.Marker([-13.453981, -56.677323], popup='<b>RIO CLARO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.988286, -43.190436], popup='<b>FORTE DE COPACABANA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.645910999999998, -42.415775], popup='<b>SILVA JARDIM</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.871321, -42.609219], popup='<b>SAQUAREMA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.496294, -46.620088], popup='<b>SAO PAULO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.358052, -49.028877], popup='<b>BAURU</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.750231, -45.603836], popup='<b>CAMPOS DO JORDÃO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.119867000000003, -51.408637], popup='<b>PRESIDENTE PRUDENTE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.584475, -47.382549], popup='<b>FRANCA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.980353, -47.883927], popup='<b>SÃO CARLOS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-24.671654, -47.545896], popup='<b>IGUAPE</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.426035, -47.585552], popup='<b>SOROCABA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.981918, -48.885818], popup='<b>ITAPEVA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.852022, -48.164817], popup='<b>S. MIG. ARCANJO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.949, -49.894546000000005], popup='<b>OURINHOS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.372832, -50.974709999999995], popup='<b>RANCHARIA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.101748999999998, -48.940996999999996], popup='<b>AVARÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.703132, -47.623317], popup='<b>PIRACICABA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.666045, -49.734854], popup='<b>LINS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.041667999999998, -45.520841], popup='<b>TAUBATÉ</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.403221, -49.966031], popup='<b>VOTUPORANGA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.165025, -50.595134], popup='<b>JALES</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.319138, -50.930158], popup='<b>VALPARAISO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.085675, -49.920388], popup='<b>JOSÉ BONIFÁCIO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.132937, -48.840426], popup='<b>ARIRANHA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.855567, -48.799717], popup='<b>IBITINGA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.78056, -47.07528], popup='<b>CASA BRANCA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.415062, -46.805292], popup='<b>ITAPIRA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.228362, -45.417055], popup='<b>S LUIS DO PARAITINGA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-22.471204999999998, -48.557573], popup='<b>BARRA BONITA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.993611, -46.256389], popup='<b>MOELA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-24.962819, -48.416379], popup='<b>BARRA DO TURVO</b>', tooltip=tooltip).add_to(m)

folium.Marker([-21.338432, -48.114007], popup='<b>PRADÓPOLIS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.559175, -48.544996999999995], popup='<b>BARRETOS</b>', tooltip=tooltip).add_to(m)

folium.Marker([-20.35976, -47.775220000000004], popup='<b>ITUVERAVA</b>', tooltip=tooltip).add_to(m)

folium.Marker([-23.523889999999998, -46.86945], popup='<b>BARUERI</b>', tooltip=tooltip).add_to(m)



m
df[df['wsnm']=='SÃO GONÇALO'].head(2)
df[df['wsnm']=='RIO CLARO'].head(2)
df.loc[df['wsnm']=='RIO CLARO','lat'] = -22.722778

df.loc[df['wsnm']=='RIO CLARO','lon'] = -44.135833

df.loc[df['wsnm']=='SÃO GONÇALO','lat'] = -22.826944

df.loc[df['wsnm']=='SÃO GONÇALO','lon'] = -43.053889
#df.to_hdf('cleaned_df.hdf','mydata',mode='w')
#df = pd.read_hdf('cleaned_df.hdf','mydata')
sns.set(rc={'figure.figsize':(20, 14)})

is_373 = df['wsid'] == 373

station_373 = df[is_373]

station_373['temp'].plot(linewidth=0.3);
sns.set(rc={'figure.figsize':(20, 14)})

is_420 = df['wsid'] == 420

station_420 = df[is_420]



station_420['temp'].plot(linewidth=0.3);

is_420 = df['wsid'] == 420

station_420 = df[is_420]

station_420_month = station_420.tail(720)

station_420_month_tmax = station_420_month['tmax'].groupby(station_420_month['da'])

station_420_month_tmin = station_420_month['tmin'].groupby(station_420_month['da'])

tmax_mean = station_420_month_tmax.mean()

tmin_mean = station_420_month_tmin.mean()

plt.figure()

tmax_mean.plot(color='red', figsize=[8,5])

tmin_mean.plot(color='blue')
index = []

year = []

city = []

tmax = []



for i in df.yr.unique():

    k = df.loc[df['yr'] == int(i)].tmax.nlargest()

    for w in range(5):

        try:

            g = df.loc[k.index[w]][df.loc[k.index[w]].tmax == k[w]]

            city.append(g.iloc[0]['city'])

            year.append(i)

            index.append(k.index[w])

            tmax.append(k[w])

        except:

            pass

        

df1 = pd.DataFrame({'date': index,'city': city,'tmax': tmax, 'year':year})

df1
index = []

city = []

tmin = []

year = []



for i in df.yr.unique():

    k = df.loc[df['yr'] == int(i)].tmin.nsmallest()

    for w in range(5):

        try:

            g = df.loc[k.index[w]][df.loc[k.index[w]].tmin == k[w]]

            city.append(g.iloc[0]['city'])

            year.append(i)

            index.append(k.index[w])

            tmin.append(k[w])

        except:

            pass

        

df2 = pd.DataFrame({'date': index,'city': city,'tmin': tmin,'year':year})

df2
group_df = df['prcp'].groupby(df['prov'])

group_df.sum()
group_df_wsid = df['prcp'].groupby(df['wsid'])

group_df_wsid.sum()
group_df_wsid.max()
import seaborn

seaborn.set_style('ticks')

%matplotlib inline
is_379 = df['wsid'] == 379

wsid_379 = df[is_379]
total_count = wsid_379.shape[0]

calm_count = wsid_379.query("wdsp == 0").shape[0]



print('Of {} total observations, {} have calm winds.'.format(total_count, calm_count))
def speed_labels(bins, units):   

    labels = []

    for left, right in zip(bins[:-1], bins[1:]):

        if left == bins[0]:

            labels.append('calm'.format(right))

        elif np.isinf(right):

            labels.append('>{} {}'.format(left, units))

        else:

            labels.append('{} - {} {}'.format(left, right, units))



    return list(labels)



def _convert_dir(directions, N=None):

    if N is None:

        N = directions.shape[0]

    barDir = directions * np.pi/180. - np.pi/N

    barWidth = 2 * np.pi / N

    return barDir, barWidth
spd_bins = [-1, 0, 3, 6, 9, 12, 15, 18, np.inf]

spd_labels = speed_labels(spd_bins, units='m/s')



dir_bins = np.arange(-7.5, 370, 15)

#print(dir_bins)

dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2
rose1 = (

    wsid_379.assign(WindSpd_bins=lambda df:

            pd.cut(df['wdsp'], bins=spd_bins, labels=spd_labels, right=True)

         )

        .assign(WindDir_bins=lambda df:

            pd.cut(df['wdct'], bins=dir_bins, labels=dir_labels, right=False)

         )

        .replace({'WindDir_bins': {360: 0}})

        .groupby(by=['WindSpd_bins', 'WindDir_bins'])

        .size()

        .unstack(level='WindSpd_bins')

        .fillna(0)

        .assign(calm=lambda df: calm_count / df.shape[0])

        .sort_index(axis=1)

        .applymap(lambda x: x / total_count * 100)

)

def wind_rose(rosedata, wind_dirs, palette=None):

    if palette is None:

        palette = seaborn.color_palette('bright', n_colors=rosedata.shape[1])



    bar_dir, bar_width = _convert_dir(wind_dirs)



    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_theta_direction('clockwise')

    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):

        if n == 0:

            # first column only

            ax.bar(bar_dir, rosedata[c1].values, 

                   width=bar_width,

                   color=palette[0],

                   edgecolor='none',

                   label=c1,

                   linewidth=0)



        # all other columns

        ax.bar(bar_dir, rosedata[c2].values, 

               width=bar_width, 

               bottom=rosedata.cumsum(axis=1)[c1].values,

               align='edge',

               color=palette[n+1],

               edgecolor='none',

               label=c2,

               linewidth=0)



    leg = ax.legend(loc=(0.75, 0.95), ncol=2)

    xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    

    return fig

directions = np.arange(0, 360, 15)

fig = wind_rose(rose1, directions)
is_375 = df['wsid'] == 375

wsid_375 = df[is_375]



rose2 = (

    wsid_375.assign(WindSpd_bins=lambda df:

            pd.cut(df['wdsp'], bins=spd_bins, labels=spd_labels, right=True)

         )

        .assign(WindDir_bins=lambda df:

            pd.cut(df['wdct'], bins=dir_bins, labels=dir_labels, right=False)

         )

        .replace({'WindDir_bins': {360: 0}})

        .groupby(by=['WindSpd_bins', 'WindDir_bins'])

        .size()

        .unstack(level='WindSpd_bins')

        .fillna(0)

        .assign(calm=lambda df: calm_count / df.shape[0])

        .sort_index(axis=1)

        .applymap(lambda x: x / total_count * 100)

)
fig = wind_rose(rose2, directions)