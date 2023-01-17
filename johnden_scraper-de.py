import requests      # Библиотека для отправки запросов
import numpy as np   # Библиотека для матриц, векторов и линала
import pandas as pd  # Библиотека для табличек
import time          # Библиотека для тайм-менеджмента
page_link = 'https://auto.ru/htmlsitemap/mark_model_catalog.html'
response = requests.get(page_link)
response
from bs4 import BeautifulSoup
page = BeautifulSoup(response.text, 'html.parser')
body = page.find('body')
links = body.find_all('a')
# Сделаем словарь с марками и моделями
dict_models = {}
for link in links:
    transit_link = str(link)[23:]
    a = transit_link.split('/')[0].upper()
    b = transit_link.split('/')[1].upper()
    if a not in dict_models:
        dict_models[a] = list()
        dict_models[a].append(b)
    else:
        dict_models[a].append(b)
url2 = 'https://auto.ru/-/ajax/desktop/listing/'
# Создадим нужные заголовки

headers = '''Host: auto.ru
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:70.0) Gecko/20100101 Firefox/70.0
Accept: */*
Accept-Language: ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3
Accept-Encoding: gzip, deflate, br
Referer: https://auto.ru/moskovskaya_oblast/cars/bmw/all/?output_type=list&page=1
x-client-app-version: 202009.04.110236
x-page-request-id: fdd6d0439fd6f805b8eb938aa1932376
x-client-date: 1599322600985
x-csrf-token: b1760c60ed4211e7e2a539ec2e2eaa86aa3e12066babbf49
x-requested-with: fetch
content-type: application/json
Origin: https://auto.ru
Content-Length: 112
Connection: keep-alive
Cookie: _csrf_token=b1760c60ed4211e7e2a539ec2e2eaa86aa3e12066babbf49; autoru_sid=a%3Ag5f51592f2m4ft261v92m47l42fs0c7f.de76e405f5554e6d59b99384cd04e1d0%7C1599166767985.604800.RClj_5rN_y7tYFld07jmoQ.gMkwnzEOGWAvc1qw8NJ8rhX3ifhAvdaLk1bxXKQKQec; autoruuid=g5f51592f2m4ft261v92m47l42fs0c7f.de76e405f5554e6d59b99384cd04e1d0; suid=e651fbd98685f6ec861083438281ce92.427a97addf23552d28e6a2ee6a04ca95; from_lifetime=1599322595978; from=direct; yuidcs=1; yuidlt=1; yandexuid=3884693151599166753; _ym_uid=1599166835569900547; _ym_d=1599322595; crookie=XeaqmOjjUO8l0uvySu5+M6Bc/7Q1wnlPxOLgIYrx+K5Lg86ZYvW00dLbh7SOsL4VsJ6GPVLb0Dew8x98+wmGZFUAURk=; cmtchd=MTU5OTE2NjgzNjY2Mw==; gids=1; cycada=SYL8uQd6uQghoUM9ZZpn1WY440Mwhf1aRme/LWLbdMc=; X-Vertis-DC=myt; _ym_visorc_22753222=b; _ym_isad=2; _ym_visorc_526680=w; _ym_visorc_148422=w; _ym_visorc_22596877=b; _ym_visorc_148383=w'''.strip().split('\n')

dict_heared = {}

for header in headers:
    key, value = header.split(': ')
    dict_heared[key] = value
# Спарсим все объявления по БМВ

url = 'https://auto.ru/-/ajax/desktop/listing/'
bmw_offers = []
for model in dict_models['BMW']:
    params = {
        "category":"cars",
        "section":"all",
        "catalog_filter":[{"mark":'BMW',"model":model}],
        "page":1,
        "geo_id":[1]}
    response = requests.post(url, json=params, headers = dict_heared)
    data = response.json()
    i = data['pagination']['total_page_count']
    for x in range(1,i+1):
        params = {
            "category":"cars",
            "section":"all",
            "catalog_filter":[{"mark":'BMW',"model":model}],
            "page":x,
            "geo_id":[1]}
        response = requests.post(url, json=params, headers = dict_heared)
        data = response.json()
        bmw_offers.extend(data['offers'])
bmw_data = pd.DataFrame(offers)

bmw_data.to_csv('bmw_data.csv')
# Запилим словарик покороче
pop_car = ['AUDI','MERCEDES','VOLKSWAGEN','VOLVO']

pop_car_dict = {}
for i in pop_car:
    pop_car_dict[i] = dict_models[i]
# # А теперь пройдемся по несколькии маркам

# url = 'https://auto.ru/-/ajax/desktop/listing/'
# all_offers = []
# for mark in list(pop_car_dict.keys()):
#     for model in pop_car_dict[mark]:
#         params = {
#             "category":"cars",
#             "section":"all",
#             "catalog_filter":[{"mark":mark,"model":model}],
#             "page":1,
#             "geo_id":[1]}
#         response = requests.post(url, json=params, headers = dict_heared)
#         data = response.json()
#         i = data['pagination']['total_page_count']
#         for x in range(1,i+1):
#             params = {
#                 "category":"cars",
#                 "section":"all",
#                 "catalog_filter":[{"mark":mark,"model":model}],
#                 "page":x,
#                 "geo_id":[1]}
#             response = requests.post(url, json=params, headers = dict_heared)
#             data = response.json()
#             all_offers.extend(data['offers'])
# Спарсим все объявления по Ауди

url = 'https://auto.ru/-/ajax/desktop/listing/'
audi_offers = []
for model in dict_models['AUDI']:
    params = {
        "category":"cars",
        "section":"all",
        "catalog_filter":[{"mark":'AUDI',"model":model}],
        "page":1,
        "geo_id":[1]}
    response = requests.post(url, json=params, headers = dict_heared)
    data = response.json()
    i = data['pagination']['total_page_count']
    for x in range(1,i+1):
        params = {
            "category":"cars",
            "section":"all",
            "catalog_filter":[{"mark":'AUDI',"model":model}],
            "page":x,
            "geo_id":[1]}
        response = requests.post(url, json=params, headers = dict_heared)
        data = response.json()
        audi_offers.extend(data['offers'])

audi_data = pd.DataFrame(audi_offers)

audi_data.to_csv('audi_data.csv')
# Спарсим все объявления по MERCEDES

url = 'https://auto.ru/-/ajax/desktop/listing/'
MERCEDES_offers = []
for model in dict_models['MERCEDES']:
    params = {
        "category":"cars",
        "section":"all",
        "catalog_filter":[{"mark":'MERCEDES',"model":model}],
        "page":1,
        "geo_id":[1]}
    response = requests.post(url, json=params, headers = dict_heared)
    data = response.json()
    i = data['pagination']['total_page_count']
    for x in range(1,i+1):
        params = {
            "category":"cars",
            "section":"all",
            "catalog_filter":[{"mark":'MERCEDES',"model":model}],
            "page":x,
            "geo_id":[1]}
        response = requests.post(url, json=params, headers = dict_heared)
        data = response.json()
        MERCEDES_offers.extend(data['offers'])
        
MERCEDES_data = pd.DataFrame(MERCEDES_offers)

MERCEDES_data.to_csv('MERCEDES_data.csv')
# Спарсим все объявления по VOLKSWAGEN

url = 'https://auto.ru/-/ajax/desktop/listing/'
VOLKSWAGEN_offers = []
for model in dict_models['VOLKSWAGEN']:
    params = {
        "category":"cars",
        "section":"all",
        "catalog_filter":[{"mark":'VOLKSWAGEN',"model":model}],
        "page":1,
        "geo_id":[1]}
    response = requests.post(url, json=params, headers = dict_heared)
    data = response.json()
    i = data['pagination']['total_page_count']
    for x in range(1,i+1):
        params = {
            "category":"cars",
            "section":"all",
            "catalog_filter":[{"mark":'VOLKSWAGEN',"model":model}],
            "page":x,
            "geo_id":[1]}
        response = requests.post(url, json=params, headers = dict_heared)
        data = response.json()
        VOLKSWAGEN_offers.extend(data['offers'])
VOLKSWAGEN_data = pd.DataFrame(VOLKSWAGEN_offers)

VOLKSWAGEN_data.to_csv('VOLKSWAGEN_data.csv')
# попробуем еще раз пройтись циклом по нескольким маркам
mark_list = ['CHEVROLET','HYUNDAI','FORD','KIA','MITSUBISHI','TOYOTA']
big_offers = []

for mark in mark_list:
    for model in dict_models[mark]:
        params = {
            "category":"cars",
            "section":"all",
            "catalog_filter":[{"mark":mark,"model":model}],
            "page":1,
            "geo_id":[1]}
        response = requests.post(url, json=params, headers = dict_heared)
        data = response.json()
        i = data['pagination']['total_page_count']
        for x in range(1,i+1):
            params = {
                "category":"cars",
                "section":"all",
                "catalog_filter":[{"mark":mark,"model":model}],
                "page":x,
                "geo_id":[1]}
            response = requests.post(url, json=params, headers = dict_heared)
            data = response.json()
            big_offers.extend(data['offers'])

big_data = pd.DataFrame(big_offers)

big_data.to_csv('big_offers.csv')
