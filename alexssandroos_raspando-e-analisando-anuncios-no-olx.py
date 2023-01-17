def spyder(url):

    response = requests.get(url)

    return BeautifulSoup(response.text, 'html5lib')



def raspa_anuncios(url):

    anuncios = []

    search = spyder(url)

    html_anuncios = search.findAll('a', {"class": "OLXad-list-link"})

    for anuncio in html_anuncios:

        dados = []

        dados.append(anuncio.get('title'))

        dados.append(anuncio.get('href'))

        dados.append(anuncio.find('p',{'class': 'OLXad-list-price'}).getText())

        try:

            dados.append(anuncio.find('p',{'class': 'OLXad-list-old-price'}).getText())

        except AttributeError as e:

            dados.append(0)

        dados.append(anuncio.find('p',{'class': 'text detail-region'}).getText())

        try:

            dados.append(anuncio.find('img',{'class': 'image'}).get('src'))

        except AttributeError as e:

            dados.append('')

        dados.append(raspa_descricao(anuncio.get('href')))

        anuncios.append(dados)

        

    return anuncios

    

def raspa_descricao(url):

    search_details = spyder(url)

    anuncio_detail = search_details.findAll('div', {"class": "page_OLXad-view"})

    return anuncio_detail[0].find('div',{'class': 'OLXad-description'}).find('p').getText()

    
import numpy as np 

import pandas as pd

import requests

import urllib.request

import time

from bs4 import BeautifulSoup
anuncios = raspa_anuncios('https://ce.olx.com.br/computadores-e-acessorios?ot=1&ps=1000&q=macbook')
DATAFRAME = pd.DataFrame(anuncios,columns=['Titulo','URL','Preco','Preco_anterior', 'Origem', 'URL_image','Descricao'] )
DATAFRAME.Origem = DATAFRAME.apply(lambda x: str(x.Origem).strip().replace('\n','').replace('\t',''), axis=1)

DATAFRAME.Descricao = DATAFRAME.apply(lambda x: str(x.Descricao).strip().replace('\n','').replace('\t',''), axis=1)

DATAFRAME.Preco = DATAFRAME.apply(lambda x: float(str(x.Preco).strip().replace('R$ ','')), axis=1)

DATAFRAME.Preco_anterior = DATAFRAME.apply(lambda x: float(str(x.Preco_anterior).strip().replace('R$ ','')), axis=1)
DATAFRAME.describe()
DATAFRAME.Preco.plot(kind='box')
DATAFRAME.groupby('Origem')['Preco'].mean().sort_values().plot(kind='barh')