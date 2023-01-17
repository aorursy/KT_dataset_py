import pandas as pd

import numpy as np

#os.chdir('/home/mario/Dropbox/cursos/python/haslwanter')

#print( os.path.abspath(os.curdir) )
from bs4 import BeautifulSoup

import requests

URL ='https://es.wikipedia.org/wiki/Anexo:Municipios_de_C%C3%B3rdoba_(Colombia)'

html_content = requests.get(URL).text

#pagina = urllib.request.urlopen(URL)

soup = BeautifulSoup(html_content, "lxml")

print(soup.title.text)
gdp_table = soup.find("table", attrs={"class": "sortable wikitable"})

#print(gdp_table)

## Para recuperar los nombres de las columnas (sin bandera) 

gdp_table_head = gdp_table.tbody.find_all("th")

#print(gdp_table_head)

headings = []

for i in range(0,len(gdp_table_head)):

    headings.append(gdp_table_head[i].text.replace('\n', ' ').strip())

#print(headings)

## Nombres de columna recuperados 



### recuperar filas 

gdp_table_data = gdp_table.tbody.find_all("tr") 



data=[]



#print(data)

for j in range(1,len(headings)):

    col=[]

    for i in range(1,len(gdp_table_data)):

        col.append(gdp_table_data[i].find_all("td")[j].text.replace('\n', ' ').strip()) 

    data.append(col)



#print(data)

data=np.array(data).T

MunCord=pd.DataFrame(data)

#print( MunCord.head() )

MunCord.columns=headings[1:]

#print( MunCord.tail(12))

#print( MunCord.info() )



# la recuperación la hace en formato string, hay que pasar las 

# numéricas a a ese tipo de datos 

MunCord['Altitud (m.s.n.m.)']=pd.to_numeric(MunCord['Altitud (m.s.n.m.)'])

#MunCord['Temperatura Promedio (°C)']=pd.to_numeric(MunCord['Temperatura Promedio (°C)'])

#MunCord['Área (km²)']=pd.to_numeric(MunCord['Área (km²)'])

#MunCord['Habitantes (2016)']=pd.to_numeric(MunCord['Habitantes (2016)'])

MunCord['Año de fundación']=pd.to_numeric(MunCord['Año de fundación'])

#MunCord.info()

#MunCord





#MunCord['Temperatura Promedio (°C)']=MunCord['Temperatura Promedio (°C)'].replace('°','')



#print( MunCord.tail(12))



tprom=[]

for cadena in MunCord['Temperatura Promedio (°C)']:

    tprom.append(cadena.replace('°',''))



tprom=pd.to_numeric(tprom)

MunCord['tprom']=tprom



area=[]

for cadena in MunCord['Área (km²)']:

    area.append(cadena.replace('.',''))



area=pd.to_numeric(area)

MunCord['area']=area



# Habitantes

habit=[]

for cadena in MunCord['Habitantes (2016)']:

    #print(cadena)

    habit.append(cadena.replace(u'\xa0', u''))

#print(habit)

habit=pd.to_numeric(habit)

MunCord['habit']=habit

MunCord 



# quitar las columnas convertidas a numericas 

MunCord2=MunCord.drop(['Temperatura Promedio (°C)','Habitantes (2016)','Área (km²)'], axis=1)



# escribir el data al disco duro 

MunCord2.to_csv('MunicipiosCord.csv')



MunCord2.head()
