# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#спарсил BMW по 2м городам (Питер и Москва). Для Питера PARAMS2 подставлял, для Мск-PARAMS
import requests  # Библиотека для запроса
import json
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
i2=0
gee: DataFrame = pd.DataFrame(
    columns=['modelDate', 'bodyType', 'fuelType', 'color', 'description', 'Custom_cleared', 'Owners_number',
             'PTS', 'VIN', 'steeringWheel', 'productionDate', 'Price_rub', 'Privod', 'vehicleTransmission', 'Salon', 'Ownership', 'Region',
             'State', 'mileage', 'Tip_auto', 'numberOfDoors', 'Class_auto', 'Name_auto', 'enginePower',
             'vehicleConfiguration', 'Model_info', 'name'])
HEADERS2 = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
    'Connection': 'keep-alive',
    'Content-Length': '137',
    'content-type': 'application/json',
    'Cookie': '_csrf_token=1c0ed592ec162073ac34d79ce511f0e50d195f763abd8c24; autoru_sid=a%3Ag5e3b198b299o5jhpv6nlk0ro4daqbpf.fa3630dbc880ea80147c661111fb3270%7C1580931467355.604800.8HnYnADZ6dSuzP1gctE0Fw.cd59AHgDSjoJxSYHCHfDUoj-f2orbR5pKj6U0ddu1G4; autoruuid=g5e3b198b299o5jhpv6nlk0ro4daqbpf.fa3630dbc880ea80147c661111fb3270; suid=48a075680eac323f3f9ad5304157467a.bc50c5bde34519f174ccdba0bd791787; from_lifetime=1580933172327; from=yandex; X-Vertis-DC=myt; crookie=bp+bI7U7P7sm6q0mpUwAgWZrbzx3jePMKp8OPHqMwu9FdPseXCTs3bUqyAjp1fRRTDJ9Z5RZEdQLKToDLIpc7dWxb90=; cmtchd=MTU4MDkzMTQ3MjU0NQ==; yandexuid=1758388111580931457; bltsr=1; navigation_promo_seen-recalls=true',
    'Host': 'auto.ru',
    'origin': 'https://auto.ru',
    'Referer': 'https://auto.ru/ryazan/cars/mercedes/all/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'x-client-app-version': '202002.03.092255',
    'x-client-date': '1580933207763',
    'x-csrf-token': '1c0ed592ec162073ac34d79ce511f0e50d195f763abd8c24',
    'x-page-request-id': '60142cd4f0c0edf51f96fd0134c6f02a',
    'x-requested-with': 'fetch'
}
import requests  # Библиотека для запроса
import json
from bs4 import BeautifulSoup

a = 1
while a <= 99:
    global modelDate, bodyType, fuelType, color, description, Custom_cleared, Owners_number, PTS, VIN, steeringWheel
    global productionDate, Price_rub, Privod, vehicleTransmission, Salon, Ownership, State, mileage, Tip_auto, numberOfDoors, Class_auto
    global Name_auto,enginePower,vehicleConfiguration, Model_info, name
    URL = 'https://auto.ru/-/ajax/desktop/listing/'
    PARAMS2={"catalog_filter": [{"mark": "BMW"}],"category": "cars","geo_id": [10174],"output_type": "list","page": a,"section": "all"}
    PARAMS = {"category":"cars","section":"all","output_type":"list","page":a,"catalog_filter":[{"mark":"BMW"}],"geo_radius":200,"geo_id":[213]}
    response = requests.post(URL, json=PARAMS2, headers=HEADERS2)
    data = response.json()['offers']
    img_url = []
    i = 0  # Переменная для перехода по объявлениям
    while i <= len(data) - 1:  # len(data)-1 это количество пришедших объявлений
        try:
            bodyType = str(data[i]['vehicle_info']['configuration']['body_type'])
        except:
            bodyType = 'No inf'

        # Категория автомобиля
        try:
            fuelType = str(data[i]['vehicle_info']['tech_param']['engine_type'])
        except:
            fuelType = 'No inf'

        # Цвет автомобиля (возвращается в формате hex)
        try:
            color =str(data[i]['color_hex'])
        except:
            color = 'No inf'

        # Описание автомобиля
        try:
            description = str(data[i]['description'])
        except:
            description = 'No inf'

        # Растаможен ли автомобиль (возвращает True или False)
        try:
            Custom_cleared = str(data[i]['documents']['custom_cleared'])
        except:
            Custom_cleared = 'Not custom cleared'

        # Лицензия на автомобиль
        try:
            modelDate = str(data[i]['vehicle_info']['super_gen']['year_from'])
        except:
            modelDate = 'No inf'

        # Колличество владельцев автомобиля
        try:
            Owners_number = str(data[i]['documents']['owners_number'])
        except:
            Owners_number = 'The number of owners is not specified'

        # PTS автомобиля
        try:
            PTS = str(data[i]['documents']['pts'])
        except:
            PTS = 'Not PTS'

        # VIN автомобиля
        try:
            VIN = str(data[i]['documents']['vin'])
        except:
            VIN = 'Not VIN'

        try:
            steeringWheel = str(data[i]['vehicle_info']['steering_wheel'])
        except:
            steeringWheel = 'No inf'

        # Год выпуска автомобиля
        try:
            productionDate = str(data[i]['documents']['year'])
        except:
            productionDate = 'No inf'

        # Цена в рублях, евро и долларах
        try:
            Price_rub = str(data[i]['price_info']['RUR'])
        except:
            Price_rub = 'Not price rub'

        try:
            Privod = str(data[i]['vehicle_info']['tech_param']['gear_type'])
        except:
            Privod = 'No inf'

        try:
            vehicleTransmission = str(data[i]['vehicle_info']['tech_param']['transmission'])
        except:
            vehicleTransmission = 'No inf'

        # С салона ли машина или нет
        try:
            Salon = str(data[i]['salon']['is_official'])
        except:
            Salon = 'Not salon'

        # Координаты места нахождения машины (возвращается долгота и широта)
        try:
            Ownership = str(data[i]['documents']['purchase_date']['year']) + '-' + str(
                data[i]['documents']['purchase_date']['month'])
        except:
            Ownership = 'No inf'

        # Регион, в котором находится автомобиль
        try:
            Region = str(data[i]['seller']['location']['region_info']['name'])
        except:
            Region = 'Not region'

        # Временная зона в которой находится автомобиль
        try:
            State =str(data[i]['state']['state_not_beaten'])
        except:
            State = 'No inf'

        # Пробег автомобиля
        try:
            mileage = str(data[i]['state']['mileage'])
        except:
            mileage = 'No inf'

        # Картинки автомобиля
        # Возвращается несколько фото, мы их добавляем в словарь img_url
        for img in data[i]['state']['image_urls']:
            img_url.append(img['sizes']['1200x900'])

        # Тип автомобиля
        try:
            Tip_auto = str(data[i]['vehicle_info']['configuration']['body_type'])
        except:
            Tip_auto = 'No inf'

        # Количество дверей у автомобиля
        try:
            numberOfDoors =  str(data[i]['vehicle_info']['configuration']['doors_count'])
        except:
            numberOfDoors = 'No inf'

        # Класс автомобиля
        try:
            Class_auto =str(data[i]['vehicle_info']['configuration']['auto_class'])
        except:
            Class_auto = 'Not class auto'

        # Название автомобиля
        try:
            Name_auto =str(data[i]['vehicle_info']['configuration']['human_name'])
        except:
            Name_auto = 'Not name auto'

        # Объем багажника автомобиля
        try:
            enginePower = str(data[i]['vehicle_info']['tech_param']['power'])
        except:
            enginePower = 'No inf'

        # Марка автомобиля
        try:
            vehicleConfiguration = str(data[i]['vehicle_info']['configuration']['body_type'])+" "+str(data[i]['vehicle_info']['tech_param']['transmission'])+' '+str(data[i]['lk_summary'])[0:3]
        except:
            vehicleConfiguration = 'No inf'

        # Модель автомобиля
        try:
            Model_info =  str(data[i]['vehicle_info']['model_info']['name'])
        except:
            Model_info = 'Not model info'

        # Информация об автомобиле
        try:
            name =  str(data[i]['vehicle_info']['tech_param']['human_name'])
        except:
            name = 'No inf'

        link_img = ''  # Переменная для ссылок
        for link_img_0 in img_url:  # Перебираем ссылки из словаря img_url, и записываем их в одну переменную текстом
            link_img += str(link_img_0) + '\n'

        # Переменные для разделения записей
        text_razdelitely1 = '================================================================================================================================='
        text_razdelitely2 = '================================================================================================================================='
        text_razdelitely3 = '=================================================================================================================================\n'

        # Переменная в которую всё записываем
        text = str(modelDate) + '\n' + str(bodyType) + '\n' + str(fuelType) + '\n' + str(
            color) + '\n' + str(description) + '\n' + str(Custom_cleared) + '\n' + \
               str(Owners_number) + '\n' + str(PTS) + '\n' + str(VIN) + '\n' + str(steeringWheel) + '\n' + str(
            productionDate) + '\n' + str(Price_rub) + '\n' + str(Privod) + '\n' + \
               str(vehicleTransmission) + '\n' + str(Salon) + '\n' + str(Ownership) + '\n' + str(Region) + '\n' + str(
            State) + '\n' + str(mileage) + '\n' + str(Tip_auto) + '\n' + \
               str(numberOfDoors) + '\n' + str(Class_auto) + '\n' + str(Name_auto) + '\n' + str(
            enginePower) + '\n' + str(vehicleConfiguration) + '\n' + str(Model_info) + '\n' + \
               str(
                   name) + '\n' + link_img + '\n' + text_razdelitely1 + '\n' + text_razdelitely2 + '\n' + text_razdelitely3

        # Записываем переменную в файл
        with open('Save_auto3(pit).txt', 'a',
                  encoding='UTF-8') as file:  # Открываем файл Save_auto1.txt (создаётся автоматически), на дозапись (ключ a)
            file.write(text)  # Записываем переменную
        i += 1 # Увеличиваем переменную страницы сайта на 1
        i2+=1
        gee.loc[i2] = [str(modelDate), str(bodyType), str(fuelType), str(color), str(description),
                      str(Custom_cleared), str(Owners_number), str(PTS), str(VIN), str(steeringWheel), str(productionDate),
                      str(Price_rub), str(Privod), str(vehicleTransmission), str(Salon), str(Ownership), str(Region),
                      str(State), str(mileage), str(Tip_auto), str(numberOfDoors), str(Class_auto), str(Name_auto),
                      str(enginePower), str(vehicleConfiguration), str(Model_info), str(name)]
    print('Page: ' + str(a))  # Выводим сообщение, какая страница записалась
    a += 1
gee.to_csv('bmw_pit.csv', index=False, header=True)
print('Successfully')  # Выводим информацию об успешном выполнении
#Подгрузил файлы из задания+два спарсенных датасета
import pandas as pd
from google.colab import files
up=files.upload()
# bmw.csv(application/vnd.ms-excel) - 14646212 bytes, last modified: n/a - 100% done
# bmw_pit.csv(application/vnd.ms-excel) - 5419753 bytes, last modified: n/a - 100% done
# sample_submission.csv(application/vnd.ms-excel) - 54869 bytes, last modified: n/a - 100% done
# test.csv(application/vnd.ms-excel) - 17062515 bytes, last modified: n/a - 100% done

# Saving bmw.csv to bmw.csv
# Saving bmw_pit.csv to bmw_pit.csv
# Saving sample_submission.csv to sample_submission.csv
# Saving test.csv to test.csv

import pandas as pd
import numpy as np
!pip install catboost

#подгрузил файлы в DataFrame. Парсил с Москвы и Питера, потому два файла sample, объединил. test также сразу с ценами объединил
y_test=pd.read_csv('test.csv')
sample1=pd.read_csv('bmw.csv')
x_test=pd.read_csv('sample_submission.csv')
sample2=pd.read_csv('bmw_pit.csv')
testo=x_test.merge(right=y_test,left_on='id',right_on='id')
sample=sample1.append(sample2)

#упорядочил столбцы по файлу test, небольшие преобразования пропусков и колонок для удобства
sample=sample[['bodyType','color','fuelType','modelDate','name','numberOfDoors','productionDate','vehicleConfiguration','vehicleTransmission','enginePower','description','mileage','Privod','steeringWheel','State','Owners_number','PTS','Custom_cleared','Ownership','Price_rub','Name_auto']]
sample=sample.replace('No inf','N/a').replace('The number of owners is not specified','N/a')
# sample[['Owners_number','enginePower','mileage','Price_rub']].astype('int') попытался сразу перевести все признаки в int,
#  по ошибке стало понятно, что там текстом прописано "Нет информации" или вроде того
sample.bodyType=sample.Name_auto

#В тесте комплектация сложно выглядит, отбросил, т.к. лень возиться, распаковывать
#Далее приведение данных к однообразному виду по принципу "в какой таблице удобнее, по такой и равняюсь"
testo=testo.rename(columns={'Привод':'Privod', 'Руль':'steeringWheel', 'Состояние':'State',
       'Владельцы':'Owners_number', 'ПТС':'PTS', 'Таможня':'Custom_cleared', 'Владение':'Ownership'})
testo=testo.drop(['Комплектация','Ownership'],axis=1)
sample=sample.drop('Ownership', axis=1)
sample=sample.replace('No inf','N/a')
sample.loc[sample.Owners_number=='N/a','Owners_number']=0
sample.loc[sample.PTS=='N/a','PTS']='ORIGINAL'
sample['engineDisplacement']=sample['vehicleConfiguration'].apply(lambda x: x[-3:])
testo.loc[testo.engineDisplacement=='Ele','engineDisplacement']='Ele LTR'
testo.engineDisplacement=testo.engineDisplacement.apply(lambda x: x[:3])
testo.enginePower=testo.enginePower.apply(lambda x: x[0:3])
#Привожу к единообразному виду 
testo.loc[testo.vehicleTransmission=='роботизированная','vehicleTransmission']="ROBOT"
testo.loc[testo.vehicleTransmission=='механическая','vehicleTransmission']="MECHANICAL"
testo.loc[testo.vehicleTransmission=='автоматическая','vehicleTransmission']='AUTOMATIC'
#везде одно значение, убираю
testo.drop(['Custom_cleared','id'],axis=1)

testo.price=testo.price.apply(lambda x: int(x))
# LEFT
# RIGHT
testo.loc[testo.steeringWheel=='Левый','steeringWheel']='LEFT'
# ORIGINAL
# DUPLICATE
testo.PTS=testo.loc[testo.PTS=='Оригинал','PTS']='ORIGINAL'
testo.PTS=testo.loc[testo.PTS=='Дубликат','PTS']='DUPLICATE'
# ALL_WHEEL_DRIVE
# REAR_DRIVE
# FORWARD_CONTROL
testo.loc[testo.Privod=='полный','Privod']='ALL_WHEEL_DRIVE'
testo.loc[testo.Privod=='задний','Privod']='REAR_DRIVE'
testo.loc[testo.Privod=='передний','Privod']='FORWARD_CONTROL'
testo=testo.drop(['State','id'],axis=1)
sample=sample.drop('State',axis=1)
# GASOLINE    
# DIESEL      
# HYBRID        
# ELECTRO  
testo.loc[testo.fuelType=='бензин','fuelType']='GASOLINE'
testo.loc[testo.fuelType=='дизель','fuelType']='DIESEL'
testo.loc[testo.fuelType=='гибрид','fuelType']='HYBRID'
testo.loc[testo.fuelType=='электро','fuelType']='ELECTRO'
testo.numberOfDoors=testo.numberOfDoors.apply(lambda x: int(x))
sample['price']=sample['Price_rub']
testo.bodyType=testo.bodyType.apply(lambda x: str(x[0].upper())+str(x[1:]))
testo.name=testo.name.apply(lambda x: x.split(' ')[0])
sample.name=sample.name.apply(lambda x: x.split(' ')[0])

# Переименование некоторых данных+генерация новых признаков из описания
import re
sample.loc[sample.bodyType.str.contains('Лифтбек', flags=re.I,regex=True),'bodyType']='Лифтбек'

testo['Excel']=0
testo['good']=0
testo['skin']=0
testo['secret']=0
testo['rust']=0
testo['winterWheels']=0
testo['nuance']=0

testo.loc[testo.description.str.contains('отличн|безупречн|идеал',flags=re.I,regex=True)].value_counts()
testo.loc[testo.description.str.contains('Отличн|отличн|безупречн|Безупречн|идеал|Идеал',flags=re.I,regex=True),'Excel']=1
testo.loc[testo.description.str.contains('хорош',flags=re.I,regex=True),'good']=1
testo.loc[testo.description.str.contains('кожа',flags=re.I,regex=True),'skin']=1


testo.loc[testo.description.str.contains('Рыжик|рыжик|скол',flags=re.I,regex=True),'rust']=1
testo.loc[testo.description.str.contains('N/a',flags=re.I,regex=True),'secret']=1
testo.loc[testo.description.str.contains('Комплект шин|комплект шин|комплект зимних шин',flags=re.I,regex=True),'winterWheels']=1
testo.loc[testo.description.str.contains('нюанс',flags=re.I,regex=True),'nuance']=1

sample['Excel']=0
sample['good']=0
sample['skin']=0
sample['secret']=0
sample['rust']=0
sample['winterWheels']=0
sample['nuance']=0

sample.loc[sample.description.str.contains('отличн|безупречн|идеал',flags=re.I,regex=True)].value_counts()
sample.loc[sample.description.str.contains('Отличн|отличн|безупречн|Безупречн|идеал|Идеал',flags=re.I,regex=True),'Excel']=1
sample.loc[sample.description.str.contains('хорош',flags=re.I,regex=True),'good']=1
sample.loc[sample.description.str.contains('кожа',flags=re.I,regex=True),'skin']=1


sample.loc[sample.description.str.contains('Рыжик|рыжик|скол',flags=re.I,regex=True),'rust']=1
sample.loc[sample.description.str.contains('N/a',flags=re.I,regex=True),'secret']=1
sample.loc[sample.description.str.contains('Комплект шин|комплект шин|комплект зимних шин',flags=re.I,regex=True),'winterWheels']=1
sample.loc[sample.description.str.contains('нюанс',flags=re.I,regex=True),'nuance']=1

#упорядочил колонки
sample=sample[['bodyType', 'color', 'fuelType', 'modelDate', 'name', 'numberOfDoors',
       'productionDate', 'vehicleConfiguration', 'vehicleTransmission',
       'engineDisplacement', 'enginePower', 'description', 'mileage', 'Privod',
       'steeringWheel', 'Owners_number', 'PTS', 'price', 'Excel', 'good',
       'skin', 'secret', 'rust', 'winterWheels', 'nuance']]

#перевел цвета из hex в человечий, мб где-то субъективно или "притянул" к цвету из датасета test
sample.loc[sample.color=='040001','color']='чёрный'
sample.loc[sample.color=='FAFBFB','color']='белый'
sample.loc[sample.color=='0000CC','color']='синий'
sample.loc[sample.color=='97948F','color']='серый'
sample.loc[sample.color=='FFC0CB','color']='розовый'
sample.loc[sample.color=='FFD600','color']='жёлтый'
sample.loc[sample.color=='4A2197','color']='фиолетовый'
sample.loc[sample.color=='200204','color']='коричневый'
sample.loc[sample.color=='EE1D19','color']='красный'
sample.loc[sample.color=='007F00','color']='зелёный'
sample.loc[sample.color=='C49648','color']='оранжевый'
sample.loc[sample.color=='660099','color']='пурпурный'
sample.loc[sample.color=='DEA522','color']='золотистый'
sample.loc[sample.color=='22A0F8','color']='голубой'
sample.loc[sample.color=='CACECB','color']='серебристый'
sample.loc[sample.color=='FF8649','color']='бежевый'

sample=sample.drop_duplicates(keep=False)
testo=testo.drop(['Custom_cleared', 'brand'],axis=1)

# #Проверка на дубликаты
# proverka=sample[testo.columns.to_list()].append(testo)
# print(len(proverka),len(proverka.drop_duplicates()))
# print(len(set(sample.description)),
#           len(set(testo.description)),
#           len(set(sample.description)-set(testo.description)),
#           len(set(sample.description)&set(testo.description)))
# len(proverka)
# len(sample)

# #проверка на дубликаты по ипасинию+ исключение дубликатов. 
# ##По описанию найдено было 45 дубликатов с файлом test, выкинуло почему-то больше, ну ок
# aga=list(set(sample.description)&set(testo.description))
# sample = sample.loc[~sample['description'].isin(aga)]
# len(sample)

sample=sample[sample.price!='Not price rub']
sample.price=sample.price.apply(lambda x: int(x))
testo.price=testo.price.apply(lambda x: int(x))

sample['Electro']=0
sample.loc[sample['engineDisplacement']=='Ele','Electro']=1
sample.loc[sample['engineDisplacement']=='Ele','engineDisplacement']=0
sample.engineDisplacement=sample.engineDisplacement.apply(lambda x: float(x))

testo['Electro']=0
testo.loc[testo['engineDisplacement']=='und','engineDisplacement']='Ele'
testo.loc[testo['engineDisplacement']=='Ele','Electro']=1
testo.loc[testo['engineDisplacement']=='Ele','engineDisplacement']=0
testo.engineDisplacement=testo.engineDisplacement.apply(lambda x: float(x))

testo.Owners_number=testo.Owners_number.apply(lambda x: x[:2])
# 3 или более    
# 1 владелец     
# 2 владельца   
testo.Owners_number=testo.Owners_number.apply(lambda x: int(x))
sample.Owners_number=sample.Owners_number.apply(lambda x: int(x))
testo.mileage=testo.mileage.apply(lambda x: int(x))
testo.modelDate=testo.modelDate.apply(lambda x: int(x))
testo.productionDate=testo.productionDate.apply(lambda x: int(x))
testo.enginePower=testo.enginePower.apply(lambda x: int(x))

#Dummy вместо category features
sdummy=pd.get_dummies(sample[['bodyType','color','fuelType','vehicleTransmission','Privod','steeringWheel','PTS']])
tdummy=pd.get_dummies(testo[['bodyType','color','fuelType','vehicleTransmission','Privod','steeringWheel','PTS']])
notinlist=list(set(sdummy)-set(tdummy))
testo[notinlist]=0

#добавил, старые убрал
testo[tdummy.columns.to_list()]=tdummy
sample[sdummy.columns.to_list()]=sdummy

sample=sample.drop(['bodyType','color','fuelType','vehicleTransmission','Privod','steeringWheel','PTS','name','description','vehicleConfiguration'],axis=1)
testo=testo.drop(['bodyType','color','fuelType','vehicleTransmission','Privod','steeringWheel','PTS','name','description','vehicleConfiguration'],axis=1)

#проверка одинаковости колонок
testo=testo[sample.columns]
sample.columns.to_list()==testo.columns.to_list()

# proverka2=sample[testo.columns.to_list()].append(testo)
# print(len(proverka2),len(proverka2.drop_duplicates(keep=False)))

# len(proverka2)
# proverka2[proverka2.duplicated(keep=False)]


#модель без перебора параметров
x=sample.drop('price',axis=1).apply(pd.to_numeric)
y=sample.price
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
xtr,xr,ytr,yr=train_test_split(x,y,test_size=0.23)
sc=StandardScaler()
xtr=sc.fit_transform(xtr)
xr=sc.transform(xr)
rf.fit(xtr,ytr)
ypr=rf.predict(xr)
from sklearn.metrics import r2_score, mean_absolute_error
r2_score(yr,ypr)
np.mean(np.abs((yr - ypr) / yr)) * 100


#c перебором
from sklearn.model_selection import RandomizedSearchCV
rf=RandomForestRegressor(random_state=42)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(xtr, ytr)
rf2=rf.set_params(**rf_random.best_params_)
rf2.fit(xtr,ytr)

#метрика
ypr=rf2.predict(xr)
from sklearn.metrics import r2_score, mean_absolute_error
print(r2_score(yr,ypr))
np.mean(np.abs((yr - ypr) / yr)) * 100
submission=pd.DataFrame(list(rf2.predict(sc.transform(x_res))),columns=['price'])
submission.to_csv('sub.csv')
from google.colab import files
dough=files.download('sub.csv')

### в обще и целом планировал склепать Minimum Viable Product
### метрика одинаковая получилась 13.36 что с базовыми настройками леса, что с перебором разных опций
### мб где-то ошибка или данные плохо обработал, почему с параметрами по умолчанию и с best_params_ абсолютно одинаковые значения ошибки-так и не понял
### можно позже будет попробовать закинуть эту кашу в стекинг, может что-то выйдет повнятнее. 