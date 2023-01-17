import pandas as pd

import requests

from bs4 import BeautifulSoup

import re

import itertools

from selenium import webdriver

import time
proyectos = 'https://congresovisible.uniandes.edu.co/proyectos-de-ley/#q=2019&page='

# lista con las url de las 123 paginas con los proyectos de ley 

proyectos_list = []

for i in [x for x in range(1,126)]:

    proyectos_list.append(proyectos + str(i))

    

def sacar_links_pdl(lista):

    browser = webdriver.Chrome(executable_path=r"C:\web_drivers\chromedriver.exe")

    browser.get(lista)

    time.sleep(2)

    html = browser.execute_script("return document.getElementsByTagName('html')[0].innerHTML")

    time.sleep(0)

    soup = BeautifulSoup(html,'lxml')

    

    elements = soup.find_all('a')

    pl_prefix = 'https://congresovisible.uniandes.edu.co/proyectos-de-ley/' 

    pl_sufix = re.findall('/proyectos-de-ley/(p.+?)\">', str(elements))

    pl_all_list = [pl_prefix + url for url in pl_sufix]

    return pl_all_list



# sacar la lista con todas las url de los proyectos de ley 

url_proyectos_ley = list(map(sacar_links_pdl,proyectos_list))



def Na_taguer(var):

            if var == []:

                var = 'NA'

            else:

                pass

            return var

def scrap_elementos(url):

       

    try: 

        get_url = requests.get(url).text

        sopa = BeautifulSoup(get_url,'lxml')

        

        r_titulo = sopa.find_all('h1')

        try:

            titulo = re.findall('\">"|“(.+?)\[', str(r_titulo))[-1]

        except:

            titulo = titulo = re.findall('\">“(.+?)\.', str(r_titulo))

        

        try:

            tagtitulo = re.findall('\[([A-Z\d].+?)\]', str(r_titulo))[-1]

                #tagtitulo = re.findall('\[([A-Z\d].+?)\]\”', str(r_titulo))[-1]

        except:

            tagtitulo = 'NA'

        del r_titulo

        

        r_estado = sopa.find('div', class_='module5').find_all('li', text='')

        estado = re.findall('\">(.+?)</a>', str(r_estado))[-1]

        proceso = re.findall('\">(.+?)</a>', str(r_estado))

        fechas_proceso = re.findall('<p>(.+?)</p>', str(r_estado))

        del r_estado

        

        try:

            r_sinopsis = sopa.find('div', class_='module7').find('p', text='')

            sinopsis = re.findall('p>(.+?)</', str(r_sinopsis))

            sinopsis = [w.replace('<br/>', ' ') for w in sinopsis][-1]

            del r_sinopsis

        except:

            sinopsis = 'NA'

        

        r_datos_grls = sopa.find('ul', class_='lista6').find_all('li', text='')

        str_datos_grls = str(r_datos_grls)

        del r_datos_grls

        

        temas = re.findall('Tema</h3><p>(.+?)</p>', str_datos_grls)

        temas = Na_taguer(temas)

        tipo = re.findall('Tipo</h3><p>(.+?)</p>', str_datos_grls)[-1]

        tipo = Na_taguer(tipo)

        

        fecha_radicacion = re.findall('Fecha de Radicación </h3><p>(.+?)</p>', str_datos_grls)[-1]

        

        tags = re.findall('#q=(.+?)\">', str_datos_grls)

        tags = [x.strip(' ') for x in tags]

        tags = Na_taguer(tags)

        

        r_autores = sopa.find('div', class_='module2')

        autores = re.findall('alt=\"([A-ZÁÉÍÓÚ].+?)\" height', str(r_autores))

        autores = Na_taguer(autores)

        del r_autores

        

        r_partido_autores = sopa.find_all('div', class_='module2')

        partido_autores1 = re.findall('\d/\">(.+?)</p>', str(r_partido_autores))

        partido_autores = re.findall('\d/\">(.+?)</a>', str(partido_autores1))

        partido_autores = Na_taguer(partido_autores)

        del r_partido_autores

        del partido_autores1

        

        url_Pdl = url

    

        #return titulo,tagtitulo,estado,autores,partido_autores,proceso,fechas_proceso,sinopsis,temas,tipo,fecha_radicacion,tags,url_Pdl



        return{'Proyecto de Ley':titulo,'Palabra Clave':tagtitulo,'Estado':estado,'Autor':autores,'Partido':partido_autores ,'Proceso':proceso,'Fechas proceso':fechas_proceso,'Resumen':sinopsis,'Tema':temas,'Tipo':tipo,'Fecha Radicación':fecha_radicacion,'Tags':tags,'link':url_Pdl}



    except: 

        

        titulo = 'link dañado'

        tagtitulo = 'link dañado'

        estado = 'link dañado'

        autores = 'link dañado'

        partido_autores = 'link dañado'

        proceso = 'link dañado'

        fechas_proceso = 'link dañado'

        sinopsis = 'link dañado'

        temas = 'link dañado'

        tipo = 'link dañado'

        fecha_radicacion = 'link dañado'

        tags = 'link dañado'

        url_Pdl = url

        

        return{'Proyecto de Ley':titulo,'Palabra Clave':tagtitulo,'Estado':estado,'Autor':autores,'Partido':partido_autores,'Proceso':proceso,'Fechas proceso':fechas_proceso,'Resumen':sinopsis,'Tema':temas,'Tipo':tipo,'Fecha Radicación':fecha_radicacion,'Tags':tags,'link':url}



Macro_Dicc = list(map(scrap_elementos,url_625_PdL))

Macro_DF = pd.DataFrame(Macro_Dicc)
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
pdl = pd.read_csv('../input/Proyectos_Ley_2018_2019.csv')
(pdl['Tipo'].value_counts() / len(pdl)).plot.bar()
(pdl['Camara'].value_counts().head(10) / len(pdl)).plot.bar()
(pdl['Estado'].value_counts().head(10) / len(pdl)).plot.bar()
pdl['cuenta'] = pdl.groupby('CC')['CC'].transform('count')
sns.kdeplot(data=pdl['cuenta'], shade=True)

plt.title('Distribución de cantidad de proyectos de ley presentados por el Congreso')
pdl.cuenta.mean()
pdl.groupby(['congresista_x','cuenta']).size().nlargest(10).plot.bar()

plt.title('Congresistas que más proyectos de ley han radicado')
pdl.groupby(['congresista_x','cuenta']).size().nsmallest(25).plot.bar()

plt.title('Congresistas que MENOS proyectos de ley han radicado')
#pdl['congresista_x'].value_counts().head(10).plot.bar()
cuenta_partido = pdl.groupby('Partido')['congresista_x'].count().reset_index().rename(

    columns={'Partido':'Partido','congresista_x' : 'congresista'})
cuenta_partido
sns.kdeplot(data=cuenta_partido['congresista'], label="Polo Democratico", shade=True)

plt.title("Comparación partidos ")

plt.legend()

pdl['party_count'] = pdl.groupby('Partido')['CC'].transform('count')
pdl['party_count'] = pdl.groupby('Partido')['CC'].transform('count')
(pdl['Partido'].value_counts().head(20) / len(pdl)).plot.bar()

plt.title('Propuestas de proyectos de ley por Partido')
sns.distplot(a=pdl['party_count'], label="Centro Democrático", kde=True)
(pdl['Tema'].value_counts().head(12) / len(pdl)).plot.bar()

plt.title('Temas más relevantes para el congreso de Colombia')
(pdl['Tema'].value_counts().nsmallest(12) / len(pdl)).plot.bar()

plt.title('Temas menos relevantes para el Congreso de Colombia')
(pdl['Tags'].value_counts().head(12) / len(pdl)).plot.bar()
(pdl['Tags'].value_counts().nsmallest(12) / len(pdl)).plot.bar()