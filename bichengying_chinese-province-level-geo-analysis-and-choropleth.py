# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

from folium import plugins



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
from urllib import request

import json

with request.urlopen('https://raw.githubusercontent.com/longwosion/geojson-map-china/master/china.json') as response:

    china_geojson = json.loads(response.read())

china_geojson['features'][31]
m = folium.Map(

    location=[35, 110],

    tiles='Stamen Terrain',

    zoom_start=4

)

folium.GeoJson(

    china_geojson,

    name='china geo',

    style_function=lambda x: {'fillColor': '#0000ff', 'fillOpacity': .2, 'weight': 2}

).add_to(m)



folium.LayerControl().add_to(m)



plugins.Fullscreen(

    position='topright',

    title='Full Screen',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(m)



m
_prov_zh_en = """北京市 Beijing

天津市 Tianjin

河北省 Hebei

山西省 Shanxi

内蒙古自治区 Inner Mongolia

辽宁省 Liaoning

吉林省 Jilin

黑龙江省 Heilongjiang

上海市 Shanghai

江苏省 Jiangsu

浙江省 Zhejiang

安徽省 Anhui

福建省 Fujian

江西省 Jiangxi

山东省 Shandong

河南省 Henan

湖北省 Hubei

湖南省 Hunan

广东省 Guangdong

广西壮族自治区 Guangxi

海南省 Hainan

四川省 Sichuan

贵州省 Guizhou

云南省 Yunnan

西藏自治区 Tibet

重庆市 Chongqing

陕西省 Shaanxi

甘肃省 Gansu

青海省 Qinghai

宁夏回族自治区 Ningxia

新疆维吾尔自治区 Xinjiang

台湾省 Taiwan

香港特别行政区 Hong Kong

澳门特别行政区 Macau"""



prov_zh2en = {}

prov_en2zh = {}

for line in _prov_zh_en.split("\n"):

    zh, en = line.split(" ", 1)

    prov_zh2en[zh] = en

    prov_en2zh[en] = zh

    



_prov_zh_code = """北京市（110000 BJ）

天津市（120000 TJ）

河北省（130000 HE）

山西省（140000 SX）

内蒙古自治区（150000 NM）

辽宁省（210000 LN）

吉林省（220000 JL）

黑龙江省（230000 HL）

上海市（310000 SH）

江苏省（320000 JS）

浙江省（330000 ZJ）

安徽省（340000 AH）

福建省（350000 FJ）

江西省（360000 JX）

山东省（370000 SD）

河南省（410000 HA）

湖北省（420000 HB）

湖南省（430000 HN）

广东省（440000 GD）

广西壮族自治区（450000 GX）

海南省（460000 HI）

四川省（510000 SC）

贵州省（520000 GZ）

云南省（530000 YN）

西藏自治区（540000 XZ）

重庆市（500000 CQ）

陕西省（610000 SN）

甘肃省（620000 GS）

青海省（630000 QH）

宁夏回族自治区（640000 NX）

新疆维吾尔自治区（650000 XJ）

台湾省（710000 TW）

香港特别行政区（810000 HK）

澳门特别行政区（820000 MO）"""



prov_zh2code = {}

prov_code2zh = {}

prov_code2en = {}

prov_en2code = {}

for line in _prov_zh_code.split("\n"):

    prov, raw_code = line.split("（", 1)

    code = raw_code[:2]

    prov_zh2code[prov] = code

    prov_code2zh[code] = prov

    prov_en2code[prov_zh2en[prov]] = code

    prov_code2en[code] = prov_zh2en[prov]

prov_en2code
chinese_data = data[data.Country.map(

    lambda x: "China" in x or "Hong Kong" in x or "Maccu" in x

              or "Taiwan" in x)].drop("Country", axis=1)

chinese_data['ProvinceCode'] = chinese_data['Province/State'].map(

    lambda x: prov_zh2code[prov_en2zh[x]])

chinese_data.head()
chinese_agg_data = (chinese_data[["Province/State", "ProvinceCode", "Confirmed"]]

                    .groupby(["Province/State", "ProvinceCode"]).agg("sum")

                    .reset_index().sort_values(by=["Confirmed"], ascending=False))

chinese_agg_data['ConfirmedLevel'] = (chinese_agg_data['Confirmed']

                                     .map(lambda x: int(np.log2(x))))

chinese_agg_data
m = folium.Map(

    location=[35, 110],

    tiles='Stamen Terrain',

    zoom_start=4

)



folium.Choropleth(

    geo_data=china_geojson,

    name='choropleth',

    data=chinese_agg_data,

    columns=['ProvinceCode', 'ConfirmedLevel'],

    key_on='feature.properties.id',

    fill_color='Reds',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name='Confirmed Level'

).add_to(m)



folium.LayerControl().add_to(m)



m
prov_city_geojson = {}

for prov_name, code in prov_zh2code.items():

    prov_en = prov_zh2en[prov_name]

    with request.urlopen(f'https://raw.githubusercontent.com/longwosion/geojson-map-china/master/geometryProvince/{code}.json') as response:

        prov_city_geojson[prov_en] = json.loads(response.read())
m = folium.Map(

    location=[33, 113],

    tiles='Stamen Terrain',

    zoom_start=6

)

folium.GeoJson(

    china_geojson,

    name='china geo',

    style_function=lambda x: {'fillColor': '#0000ff', 'fillOpacity': .2, 'weight': 2}

).add_to(m)

folium.GeoJson(

    prov_city_geojson['Hubei'],

    name='Hubei geo',

    style_function=lambda x: {'fillColor': '#ff5511', 'fillOpacity': 0.8, 'weight': 1}

).add_to(m)



folium.LayerControl().add_to(m)



plugins.Fullscreen(

    position='topright',

    title='Full Screen',

    title_cancel='Exit me',

    force_separate_button=True

).add_to(m)



m