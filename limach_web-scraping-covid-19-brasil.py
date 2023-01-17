#Importando Pandas
import pandas as pd
#Upload da base de dados Covid 19 from GitHub
GPS = 'https://raw.githubusercontent.com/wcota/covid19br/master/gps_cities.csv'
UF_dataserie= 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
Municipios = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities.csv'
UF = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-total.csv'
# Salvando o dados em Dataframes
gps = pd.read_csv(GPS)
uf_dataserie = pd.read_csv(UF_dataserie)
municipios = pd.read_csv(Municipios)
uf = pd.read_csv(UF)
# Verificando databeses
gps.head(5)
uf_dataserie = uf_dataserie[uf_dataserie.state != 'TOTAL']
uf_dataserie['date']=pd.to_datetime(uf_dataserie.date)
uf_dataserie.dtypes
municipios.tail(5)
uf = uf[uf.state !='TOTAL']
uf = uf.drop(['deathsMS'], axis=1)
# Salvando em CSV
gps.head()
gps = gps.rename(columns={"longName": "name"}).head()
gps = gps.drop(['ibgeID'], axis=1)
gps.head()
uf_dataserie.tail()
uf_dataserie = uf_dataserie.drop(['newDeaths','deathsMS','totalCasesMS'], axis=1)

municipios.head()
uf.head()
gps.to_csv(r'E:\Python\Covid\Database\Brazil\Gps.csv', index=0, sep=';')
uf_dataserie.to_csv(r'E:\Python\Covid\Database\Brazil\Uf Dataserie.csv', index=0, sep=';')
municipios.to_csv(r'E:\Python\Covid\Database\Brazil\Municipios.csv', index=0, sep=';')
uf.to_csv(r'E:\Python\Covid\Database\Brazil\UF.csv', index=0, sep=';')
