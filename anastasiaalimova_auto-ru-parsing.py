import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pprint
import requests
import time
import json

pp = pprint.PrettyPrinter()

s = requests.Session()

# делаем запрос, получаем куки, сохраняем токен
s.get('https://auto.ru')  
csrf_token = s.cookies.get("_csrf_token") 

headers = {"x-csrf-token": csrf_token}

# указываем параметры: mark - BMW, geo_id - 1 (Москва), количество машин на странице -100
data = {"catalog_filter": [{"mark": "BMW"}], "section": "all", "category": "cars", "output_type": "list", "geo_id": [1], "page_size": 100}

response = s.post('https://auto.ru/-/ajax/desktop/listing/', json=data, headers=headers).json()

pp.pprint(response)

# сохраняем в переменную общее число страниц, чтобы проитерироваться по всем страничкам и собрать машины в общий список "cars"
page_amount = response["pagination"]['total_page_count']

# одна страница уже получена
cars = response["offers"]

for i in range(1, page_amount):
    data["page"] = i + 1
    cars.extend(s.post('https://auto.ru/-/ajax/desktop/listing/', json=data, headers=headers).json().get("offers", []))
    time.sleep(5)  # sleep нужен, чтобы нас не приняли за бота

len(cars)
auto_ru_cars = []
for car_dict in cars:
    auto_ru_cars.append({
        'body_type': car_dict['vehicle_info'].get('configuration', {}).get('body_type'),
        'brand': 'BMW',
        'color': car_dict.get('color_hex'),
        'fuelType': car_dict['vehicle_info'].get('tech_param', {}).get('engine_type'),
        'modelDate': 0,
        'name': "{} {}".format(car_dict['vehicle_info'].get('tech_param', {}).get('nameplate'), car_dict.get('lk_summary').split(',')[0]), 
        'numberOfDoors': car_dict['vehicle_info'].get('configuration', {}).get('doors_count'),
        'productionDate': car_dict['documents'].get('year'),
        # не стала добавлять vehicleConfiguration, так как признаки body type,transmission и engineDisplacement уже представлены отдельно
        'vehicleTransmission': car_dict['vehicle_info'].get('tech_param', {}).get('transmission'),
        'engineDisplacement': car_dict.get('lk_summary').split(' ')[0],
        'enginePower': car_dict['owner_expenses']['transport_tax'].get('horse_power'),
        'description': car_dict.get('description'),
        'mileage': car_dict.get('state', {}).get('mileage'),
        'Комплектация': car_dict.get('vehicle_info', {}).get('complectation', {}).get('available_options'),
        'Привод': car_dict['vehicle_info'].get('tech_param', {}).get('gear_type'),
        'Руль': car_dict['vehicle_info'].get('steering_wheel'),
        # если true - машина не битая
        'Состояние': car_dict.get('state', {}).get('state_not_beaten'),
        'Владельцы': car_dict['documents'].get('owners_number'),
        'ПТС': car_dict['documents'].get('pts'),
        'таможня': car_dict['documents'].get('custom_cleared'),
        'Владение': car_dict.get('owner_expenses', {}).get('transport_tax', {}).get('holding_period_month'),
        'id': car_dict['id'],
        'price': car_dict.get('price_info', {}).get('price')
    })
    
# Сохраняем в json
with open('auto_ru_car.json', 'w') as outfile:
    json.dump(auto_ru_cars, outfile)
    
# Превращаем в датасет  
with open('auto_ru_car.json') as f:
    auto_ru_cars = json.load(f)

auto_ru_df = pd.DataFrame(auto_ru_cars)
auto_ru_df.head()