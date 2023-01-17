from bs4 import BeautifulSoup

import requests, re

import pandas as pd

from io import StringIO
url = 'https://www.juntadeandalucia.es/institutodeestadisticaycartografia/badea/stpivot/stpivot/STPivot.jsp?idIframe=stpivot_frame&defaultContext=&contextOrg=/institutodeestadisticaycartografia/badea&codConsulta=38228&historyBack=&idTituloActividad=tituloActividad'
htmlPage = requests.get(url).text

soup = BeautifulSoup(htmlPage, 'html.parser')

exportTXT = soup.find(alt='CSV')['onclick']
url_base_csv = 'https://www.juntadeandalucia.es/institutodeestadisticaycartografia/badea/stpivot/stpivot/'

regex = re.compile('Print.*38228')

export = regex.findall(exportTXT)[0]
csv = requests.get(url_base_csv + export).text
dmy_dateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y")

data = StringIO(csv)

df = pd.read_csv(data, sep=';', index_col=0, parse_dates=True, date_parser=dmy_dateparser)

df = df.drop(['Unnamed: 4'],axis=1)

df = df.fillna(0)
df.head(15)
pivot = pd.pivot_table(df, values='Valor', index=['Fecha','Territorio'], columns=['Medida'])
pivot.tail(20)