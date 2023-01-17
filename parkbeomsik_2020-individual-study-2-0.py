# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import matplotlib.pyplot as plt



import folium 

from folium import plugins

from folium import FeatureGroup, LayerControl, Map, Marker

# from folium.plugins import HeatMap



import json 

from datetime import datetime



import warnings

warnings.filterwarnings("ignore")



import datetime



import plotly.graph_objects as go

import plotly.express as px



from pyproj import Proj, transform
TimeGender = pd.read_csv('../input/coronavirusdataset/TimeGender.csv')

Case = pd.read_csv('../input/coronavirusdataset/Case.csv')

Region = pd.read_csv('../input/coronavirusdataset/Region.csv')

TimeProvince = pd.read_csv('../input/coronavirusdataset/TimeProvince.csv')

SearchTrend = pd.read_csv('../input/coronavirusdataset/SearchTrend.csv')

PatientRoute = pd.read_csv('../input/coronavirusdataset/PatientRoute.csv')

SeoulFloating = pd.read_csv('../input/coronavirusdataset/SeoulFloating.csv')

Time = pd.read_csv('../input/coronavirusdataset/Time.csv')

PatientInfo = pd.read_csv('../input/coronavirusdataset/PatientInfo.csv')

Weather = pd.read_csv('../input/coronavirusdataset/Weather.csv')

TimeAge = pd.read_csv('../input/coronavirusdataset/TimeAge.csv')

Policy = pd.read_csv('../input/coronavirusdataset/Policy.csv')
PatientInfo[pd.notna(PatientInfo['contact_number'])]

PatientRoute.merge(PatientInfo, on=['patient_id']).groupby(['patient_id']).count()

PatientInfo.groupby(['infected_by']).count().sort_values('patient_id', ascending=False).head(20).style.format("{:.0f}")





# PatientInfo['age'] = 2020 - PatientInfo['birth_year'].astype(int) + 1

PatientInfo['age'] = PatientInfo['age'].str.slice(0, -1).astype(float)

PatientInfo['age_group'] = PatientInfo['age'] // 10

PatientInfo['age_group'] = [str(a).replace('.','') for a in PatientInfo['age_group']]

PatientInfo['age_gender'] = PatientInfo['age_group'] + '_' + PatientInfo['sex']



PatientInfo[PatientInfo['contact_number'] == '-'] = np.nan

PatientInfo['contact_number'] = PatientInfo['contact_number'].astype(float)



fig = plt.gcf()

fig.set_size_inches(15, 5)



classes = np.sort(pd.unique(PatientInfo['age_gender'].dropna().values.ravel()))

boxplot = sns.boxplot(x="age_gender", y="contact_number", data=PatientInfo[PatientInfo['contact_number'] != '-'][PatientInfo['contact_number'].astype(float) < 200], order = classes)

boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45)

plt.title("Age vs Contact_number")

plt.show()



print(np.sort(pd.unique(PatientInfo['age_group'].dropna().values.ravel())))
PatientRoute['date'] = pd.to_datetime(PatientRoute['date'])
Policy['start_week'] = pd.to_datetime(Policy['start_date']).dt.weekofyear



def mark_policy(fig):

    fig.update_layout(



        annotations=[

            dict(

                x=Policy[Policy['type'] == 'Alert'].iloc[1]['start_week'],

                y=1,

                xref="x",

                yref="y",

                text=f"Alert {Policy[Policy['type'] == 'Alert'].iloc[1]['detail']}",

                showarrow=True,

                arrowhead=7,

                ax=0,

                ay=-40,

                bgcolor='white'

            ),

            dict(

                x=Policy[Policy['type'] == 'Alert'].iloc[2]['start_week'],

                y=1,

                xref="x",

                yref="y",

                text=f"Alert {Policy[Policy['type'] == 'Alert'].iloc[2]['detail']}",

                showarrow=True,

                arrowhead=7,

                ax=0,

                ay=-60,

                bgcolor='white'

            ),

            dict(

                x=Policy[Policy['type'] == 'Alert'].iloc[3]['start_week'],

                y=1,

                xref="x",

                yref="y",

                text=f"Alert {Policy[Policy['type'] == 'Alert'].iloc[3]['detail']}",

                showarrow=True,

                arrowhead=7,

                ax=0,

                ay=-20,

                bgcolor='white'

            ),

            dict(

                x=Policy[Policy['type'] == 'Social'].iloc[0]['start_week'],

                y=1,

                xref="x",

                yref="y",

                text=f"{Policy[Policy['type'] == 'Social'].iloc[0]['gov_policy']}",

                showarrow=True,

                arrowhead=7,

                ax=0,

                ay=-40,

                bgcolor='white'

            )

        ]

    )    
PatientRoute['week'] = PatientRoute['date'].dt.weekofyear
PatientRoute['type'].unique()
PatientInfoRoute = PatientInfo.merge(PatientRoute, on="patient_id")



for group_name, group in PatientInfoRoute.groupby("patient_id"):

    x = 0

    y = 0

    for row_index, row in group.iterrows():

        if x == 0:

            x = row['longitude']

            y = row['latitude']

        PatientInfoRoute.loc[row_index, 'relative_x'] = PatientInfoRoute.loc[row_index, 'longitude'] - x

        PatientInfoRoute.loc[row_index, 'relative_y'] = PatientInfoRoute.loc[row_index, 'latitude'] - y
f = go.FigureWidget()

scatt = f.add_scatter()
df = PatientInfoRoute.groupby('patient_id').filter(lambda x: x['relative_x'].count()>1)

df[df['patient_id']==1700000020]['date']
from ipywidgets import interact

f = go.FigureWidget()

# for group_name, group in PatientInfoRoute.groupby("patient_id"):

#     f.add_scatter(x=group['relative_x'], y=group['relative_y'], visible=False)



# steps = []

# for i in range(len(f.data)):

#     step = dict(

#         method="update",

#         args=[{"visible": [False] * len(f.data)},

#              {"title": "Slider switched to patient_id: "}],

#     )

#     step["args"][0]["visible"][i] = True

#     steps.append(step)



# sliders = [dict(

#     active=10,

#     currentvalue={"prefix": "patient_id: "},

#     steps=steps

# )]

# f.update_layout(sliders=sliders)

f.update_xaxes(range=[-1,1])

f.update_yaxes(range=[-1.3,1.3])

f.update_layout(width=400, height=400)



scatt = f.add_scatter()



@interact(patient_id=PatientInfoRoute.groupby('patient_id').filter(lambda x: x['relative_x'].count()>1)['patient_id'].unique())

def update(patient_id="1000000001"):

    with f.batch_update():

        f.data = []

        f.add_scatter(x=PatientInfoRoute[PatientInfoRoute['patient_id'] == patient_id]['relative_x'],

                    y= PatientInfoRoute[PatientInfoRoute['patient_id'] == patient_id]['relative_y'],

                     mode='lines+markers')

        x_max = PatientInfoRoute[PatientInfoRoute['patient_id'] == patient_id]['relative_x'].abs().max()

        y_max = PatientInfoRoute[PatientInfoRoute['patient_id'] == patient_id]['relative_y'].abs().max()

        f.update_xaxes(range=[-max(x_max,y_max), max(x_max,y_max)])

        f.update_yaxes(range=[-max(x_max,y_max), max(x_max,y_max)])

        

f
# #-*- coding:euc-kr -*-

import csv, sqlite3



con = sqlite3.connect(":memory:")

cur = con.cursor()

# cur.execute("CREATE TABLE t (시군구코드,출입구일련번호,시도명,시군구명,읍면도명,도로명,건물본번,건물부번,건물명,도로명코드,건물용도분류,X좌표,Y좌표);") # use your column names here



# import os

# for dirname, _, filenames in os.walk('../input/geolocationdb/202004_위치정보요약DB_전체분'):

#     for filename in filenames:

#         if 'txt' not in filename:

#             continue

#         with open(os.path.join(dirname, filename),'rt',encoding='cp949') as fin: # `with` statement available in 2.5+

#             # csv.DictReader uses first line in file for column headings by default

#             dr = csv.DictReader(fin, delimiter="|", fieldnames=['시군구코드','출입구일련번호','법정동코드','시도명','시군구명',

#                                                                '읍면동명','도로명코드','도로명','지하여부','건물본번',

#                                                                '건물부번','건물명','우편번호','건물용도분류','건물군여부',

#                                                                '관할행정동','X좌표','Y좌표']) # comma is default delimiter

#             to_db = [(i['시군구코드'], i['출입구일련번호'], i['시도명'], i['시군구명'], i['읍면동명'], i['도로명'], 

#                       i['건물본번'], i['건물부번'], i['건물명'], i['도로명코드'], i['건물용도분류'], i['X좌표'], i['Y좌표']) for i in dr]



#         cur.executemany("""INSERT INTO t (시군구코드,출입구일련번호,시도명,시군구명,읍면도명,도로명,건물본번,건물부번,건물명,도로명코드,건물용도분류,X좌표,Y좌표) VALUES

#                         (?,?,?,?,?,?,?,?,?,?,?,?,?);""", to_db)

#         con.commit()

# cur.execute("select count() from t")

# rows = cur.fetchall()

# for row in rows:

#     print(row)
# #-*- coding:euc-kr -*- 

# import csv, sqlite3

# try:

#     cur.execute("DROP TABLE address_to_building;")

# except:

#     pass

# cur.execute("CREATE TABLE address_to_building (관리번호,도로명코드,건물본번,건물부번);") # use your column names here



# import os

# for dirname, _, filenames in os.walk('../input/geolocationdb/'):

#     for filename in filenames:

#         if 'Address_DB.txt' != filename:

#             continue

#         with open(os.path.join(dirname, filename),'rt',encoding='utf-16 LE') as fin: # `with` statement available in 2.5+

#             # csv.DictReader uses first line in file for column headings by default

#             dr = csv.DictReader(fin, delimiter="|", fieldnames=['관리번호', '도로명코드', '읍면동일련번호', '지하여부', '건물본번',

#                                                                '건물부번', '기초구역번호', '변경사유코드', '고시일자', '변경전 도로명주소',

#                                                                '상세주소 부여여부']) # comma is default delimiter

#             to_db = [(i['관리번호'], i['도로명코드'], i['건물본번'], i['건물부번']) for i in dr]



#         cur.executemany("""INSERT INTO address_to_building (관리번호,도로명코드,건물본번,건물부번) VALUES

#                         (?,?,?,?);""", to_db)

#         con.commit()

# cur.execute("select count() from address_to_building")

# rows = cur.fetchall()

# for row in rows:

#     print(row)
# UTM-K

proj_UTMK = Proj(init='epsg:5178')



#WGS1984

proj_WGS84 = Proj(init='epsg:4326')
# import requests, json



# url = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc?"

# clientId = "7233es493g"

# clientSecret = "LB06InzJDfD3q2n0PyeFMWNi3DNh7LhAfkzHGRcY"

# queryString = "coords=126.7156325,37.6152464&output=json&orders=roadaddr&sourcecrs=epsg:4326"

# header = {

#     "X-NCP-APIGW-API-KEY-ID":clientId,

#     "X-NCP-APIGW-API-KEY":clientSecret

# }



# for index,row in PatientInfoRoute.iterrows():

    

#     queryString = f"coords={row['longitude']},{row['latitude']}&output=json&orders=roadaddr&sourcecrs=epsg:4326"

#     r = requests.get(url + queryString, headers=header)

#     parsed = json.loads(r.text)

#     print(row['longitude'], row['latitude'], len(parsed['results']))

#     if len(parsed['results']) == 0:

#         print(json.dumps(parsed, indent=4, sort_keys=True))

#         print(row['type'])

#     else:

#         PatientInfoRoute.loc[index, 'new_address']=f"""{parsed['results'][0]['region']['area1']['name']}|{parsed['results'][0]['region']['area2']['name']}|

# {parsed['results'][0]['region']['area3']['name']}|

# {parsed['results'][0]['land']['name']}|

# {parsed['results'][0]['land']['number1']}|

# {'0' if not parsed['results'][0]['land']['number2'] else parsed['results'][0]['land']['number2']}"""

    

#         cur.execute("""SELECT 시도명,시군구명,읍면도명,도로명,건물본번,건물부번,건물명,도로명코드,건물용도분류

#                         FROM t 

#                         WHERE 시도명=? and 시군구명=? and 읍면도명=? and 도로명=? and 건물본번=? and 건물부번=?;""", 

#                     (parsed['results'][0]['region']['area1']['name'],

#                     parsed['results'][0]['region']['area2']['name'],

#                     parsed['results'][0]['region']['area3']['name'],

#                     parsed['results'][0]['land']['name'],

#                     parsed['results'][0]['land']['number1'],

#                      '0' if not parsed['results'][0]['land']['number2'] else parsed['results'][0]['land']['number2'],

#                     ))

#         rows = cur.fetchall()

#         print(rows)

#         if rows == []:

#             print(row['type'])

#             print('No address')

#         else:

#             print(row['type'], rows[0][-1])

#             PatientInfoRoute.loc[index, 'new_type'] = rows[0][-1]

#             PatientInfoRoute.loc[index, 'address_code'] = rows[0][-2]

#             print(f"{index}/{PatientInfoRoute.shape[0]}")

# PatientInfoRoute.to_pickle('PatientInfoRoute.pkl')

PatientInfoRoute = pd.read_pickle('../input/cached/PatientInfoRoute.pkl')
def large_type(x):

    if x['type'] in ['academy', 'school', 'university']:

        return 'education'

    elif x['type'] in ['airport', 'public_transportation', 'gas_station']:

        return 'transportation'

    elif x['type'] in ['hospital', 'pharmacy']:

        return 'medicine'

    elif x['type'] in ['store', 'restaurant', 'beauty_salon', 'bank', 'bakery', 'real_estate_agency', 'posr_office', 'lodging']:

        return 'life'

    elif x['type'] in ['pc_cafe', 'bar', 'gym', 'cafe']:

        return 'entertainment'

    elif x['type'] in ['church']:

        return 'church'

    elif x['type'] in ['etc']:

        if pd.isna(x['new_type']):

            return 'etc'

        if x['new_type'].split(',')[0] in ['주택', '숙박시설']:

            return 'house'

        elif x['new_type'].split(',')[0] in ['종교시설']:

            return 'church'

        elif x['new_type'].split(',')[0] in ['근린생활시설']:

            return 'life'        

        elif x['new_type'].split(',')[0] in ['업무시설', '공장/창고시설', '농축수산시설', '공공용시설']:

            return 'work'

        elif x['new_type'].split(',')[0] in ['유흥/위락시설', '문화/관광/레저시설']:

            return 'entertainment'

        elif x['new_type'].split(',')[0] in ['의료시설', '교육및복지시설']:

            return 'medicine'

        elif x['new_type'].split(',')[0] in ['자동차관련시설', '유통시설']:

            return 'transportation'        

        else:

            return 'etc'

    else:

        return 'etc'



PatientInfoRoute['large_type'] = PatientInfoRoute.apply(large_type, axis=1)



type_by_time = PatientInfoRoute.groupby(['week', 'large_type']).size().unstack().fillna(0)

# type_by_time = type_by_time.div(type_by_time.sum(axis=1), axis=0) * 100

type_by_time

type_by_time_age = []

# df = PatientRoute.merge(PatientInfo, on='patient_id')

df = PatientInfoRoute

for age in ['20','30','40','50','60','70','80']:

    

    new_type_by_time = df[df['age_group'] == age].groupby(['week', 'large_type']).size().unstack().fillna(0)

    if 'education' not in new_type_by_time.columns:

        new_type_by_time['education'] = 0

    if 'entertainment' not in new_type_by_time.columns:

        new_type_by_time['entertainment'] = 0

    if 'etc' not in new_type_by_time.columns:

        new_type_by_time['etc'] = 0    

    if 'house' not in new_type_by_time.columns:

        new_type_by_time['house'] = 0            

    type_by_time_age.append(new_type_by_time)
colors = px.colors.qualitative.Light24

x = type_by_time.index.tolist()

categories = ['medicine', 'transportation', 'life', 'entertainment', 'education', 'church', 'house', 'work', 'etc']

fig = go.Figure()



for i, cat in enumerate(categories):

    fig.add_trace(go.Scatter(x=x, y=type_by_time[cat],

                        hoverinfo='x+y',

                        mode='lines',

                        line=dict(width=0.5, color=colors[i]),

                        name=cat,

                        stackgroup='one',

                        groupnorm='percent'))

for age, df in enumerate(type_by_time_age):

    for i, cat in enumerate(categories):

        fig.add_trace(go.Scatter(x=df.index.tolist(), y=df[cat],

                            hoverinfo='x+y',

                            mode='lines',

                            line=dict(width=0.5, color=colors[i]),

                            name=cat,

                            stackgroup=age,

                            groupnorm='percent',

                            visible=False))    



fig.update_layout(

    title='Where most patients visited?',

    showlegend=True,

    xaxis=dict(

        range=[4, 19],

        ticksuffix=' week'

    ),

    yaxis=dict(

        type='linear',

        range=[1, 100],

        ticksuffix='%'))

mark_policy(fig)



menus = []

for i, name in enumerate(['All', '20','30','40','50','60','70','80']):

    d = dict(label=name,

                     method="update",

                     args=[{"visible": [False]*i*9 + [True]*9 + [False]*(8-i-1)*9},

                           {"title": f"Where most patients visited? (Age: {name})"}])

    menus.append(d)



fig.update_layout(

    updatemenus=[

        dict(

            type="buttons",

            direction="right",

            active=0,

            x=1,

            y=1.2,

            buttons=menus,

        )

    ],

    xaxis_title="weeks",

    yaxis_title="% in group of patients",

)



fig.show()
import pprint



colors = px.colors.qualitative.Light24

print(colors)

x=['10', '20', '30', '40', '50', '60', '70', '80']

y=['work', 'house', 'church', 'education', 'entertainment', 'life', 'transportation', 'medicine']

y.reverse()



y_list_list = []

for p_type in y[:]:

    y_list = []

    for age_group in x:

        y_list.append(len(

            PatientInfoRoute[(PatientInfoRoute['age_group'] == age_group) & (PatientInfoRoute['large_type'] == p_type)]))   

    y_list_list.append(y_list)

    

for j in range(len(y_list_list[0])):

    column = [row[j] for row in y_list_list]

    sum_column = sum(column)

    for i in range(len(y_list_list)):

        y_list_list[i][j] = y_list_list[i][j]/float(sum_column) * 100



fig = go.Figure(go.Bar(x=x, y=y_list_list[0], name='medicine', marker_color=colors[0], opacity=0.5))

for i, p_type in enumerate(y[1:]):

    fig.add_trace(go.Bar(x=x, y=y_list_list[i+1], name=p_type, marker_color=colors[i+1], opacity=0.5))





fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 

                                          'categoryarray':['10', '20', '30', '40', '50', '60', '70', '80'],

                                         'ticksuffix':'s'},

                      yaxis=dict(

                                type='linear',

                                ticksuffix='%'),

                  xaxis_title="Age group",

                    yaxis_title="% in group of patients",)

fig.show()
# succ = 0

# fail = 0



# for index,row in PatientInfoRoute.iterrows():

#     if pd.isna(row['new_address']):

#         continue

#     cur.execute("""SELECT 관리번호

#                     FROM address_to_building

#                     WHERE 도로명코드=? and 건물본번=? and 건물부번=?;""", 

#                 (row['address_code'], row['new_address'].split('|')[-2][1:], row['new_address'].split('|')[-1][1:]))

#     rows = cur.fetchall()

# #     print(rows)

#     if rows == []:

#         print(f"Failed for {(row['address_code'], row['new_address'].split('|')[-2], row['new_address'].split('|')[-1])}")

#         fail += 1

#     else:

# #         print(len(rows), rows)

#         PatientInfoRoute.loc[index, 'building_code'] = '|'.join([row[0] for row in rows])

# #         print(f"{index}/{PatientInfoRoute.shape[0]}")

#         succ += 1

# #     print(f"success for {succ}/{succ+fail}")

# from collections import defaultdict

# succ = 0

# fail = 0



# road_dict = dict()



# for index,pd_row in PatientInfoRoute.iterrows():

#     if pd.isna(pd_row['new_address']):

#         continue

#     cur.execute("""SELECT 관리번호

#                     FROM address_to_building

#                     WHERE 도로명코드=?;""", 

#                 (pd_row['address_code'],))

#     rows = cur.fetchall()

# #     print(rows)

#     if rows == []:

#         print(f"Failed for {(pd_row['address_code'], pd_row['new_address'].split('|')[-2], pd_row['new_address'].split('|')[-1])}")

#         fail += 1

#     else:

# #         print(len(rows), rows)

#         for row in rows:

#             road_dict[row[0]] = pd_row['address_code']

# #         road_dict[row['address_code']].append([row[0] for row in rows])

# #         print(f"{index}/{PatientInfoRoute.shape[0]}")

#         succ += 1

#         print(f"success for {succ}/{succ+fail}")



road_dict = pd.read_pickle('/kaggle/input/cached/road_dict.pkl')
import pickle

with open('road_dict.pkl', 'wb') as f:

    pickle.dump(road_dict, f)
PatientInfoRoute = pd.read_pickle('../input/cached/PatientInfoRoute.pkl')
if os.path.exists('/kaggle/input/cached/building_info.pkl'):

    building_info = pd.read_pickle('/kaggle/input/cached/building_info.pkl')

else:

    !pip install dbfread

    from dbfread import DBF

    import gc



    succ = 0

    fail = 0

    building_code_dict = dict()

    building_codes = set()

    for index,row in PatientInfoRoute.iterrows():

        if row['province_y'] not in building_code_dict:

            building_code_dict[row['province_y']] = set()

        if pd.isna(row['building_code']):

            continue



        building_codes.add(row['building_code'])

    building_info = pd.DataFrame(data=[], columns=['UFID','BLD_NM','DONG_NM','GRND_FLR','UGRND_FLR','PNU','ARCHAREA',

                                         'TOTALAREA','PLATAREA','HEIGHT','STRCT_CD','USABILITY','BC_RAT',

                                         'VL_RAT','BLDRGST_PK','USEAPR_DAY','REGIST_DAY','GB_CD','VIOL_BD_YN',

                                         'GEOIDN','BLDG_PNU','BLDG_PNU_Y','BLD_UNLICE','BD_MGT_SN','SGG_OID',

                                         'COL_ADM_SE'])    

    print(building_code_dict.keys())

    print("step 1")

    data = dict()



    for dirname, _, filenames in os.walk('/kaggle/input/f-fac-building/'):    

        for filename in filenames:

            city = filename.split('_')[3]

            if city not in building_code_dict:

                print(f'{city} failed')

                continue

            search_list = building_code_dict[city]

            for record in DBF(os.path.join(dirname, filename), encoding='cp949'):

                if record['BD_MGT_SN'] in building_codes:

                    for key in record.keys():

                        record[key] = [record[key]]

                    building_info = building_info.append(pd.DataFrame(data=dict(record), columns=['UFID','BLD_NM','DONG_NM','GRND_FLR','UGRND_FLR','PNU','ARCHAREA',

                                         'TOTALAREA','PLATAREA','HEIGHT','STRCT_CD','USABILITY','BC_RAT',

                                         'VL_RAT','BLDRGST_PK','USEAPR_DAY','REGIST_DAY','GB_CD','VIOL_BD_YN',

                                         'GEOIDN','BLDG_PNU','BLDG_PNU_Y','BLD_UNLICE','BD_MGT_SN','SGG_OID',

                                         'COL_ADM_SE']))



                    building_codes.remove(record['BD_MGT_SN'][0])

                    succ += 1



            print(f'{filename} finished')

            gc.collect()



    building_info.to_pickle('building_info.pkl')
# !pip install dbfread

# from dbfread import DBF

# import gc



# from collections import defaultdict

# vl_rat_dict = defaultdict(list)



# for dirname, _, filenames in os.walk('/kaggle/input/f-fac-building/'):    

#     for filename in filenames:

#         city = filename.split('_')[3]

#         for record in DBF(os.path.join(dirname, filename), encoding='cp949'):

#             if record['BD_MGT_SN'] in road_dict.keys():

#                 vl_rat_dict[road_dict[record['BD_MGT_SN']]].append(float(record['VL_RAT']))



#         print(f'{filename} finished')

#         gc.collect()

# print("Step 2")

# vl_rat_avg_dict = dict() 

# for adr_code, vl_rat_list in vl_rat_dict.items():

#     vl_rat_avg_dict[adr_code] = sum(vl_rat_list)/len(vl_rat_list)

    

vl_rat_avg_dict = pd.read_pickle('/kaggle/input/cached/vl_rat_avg_dict.pkl')
import pickle

with open('vl_rat_avg_dict.pkl', 'wb') as f:

    pickle.dump(vl_rat_avg_dict, f)
print("Step 3")

for index,pd_row in PatientInfoRoute.iterrows():

    try:

        PatientInfoRoute.loc[index, 'vl_rat_average'] = vl_rat_avg_dict[pd_row['address_code']]

    except:

        pass

    

PatientInfoRoute['BD_MGT_SN'] = PatientInfoRoute['building_code']



PatientInfoRouteBuilding = PatientInfoRoute.merge(building_info, on='BD_MGT_SN')

# PatientInfoRouteBuilding['건폐율'] = PatientInfoRouteBuilding['ARCHAREA'] / PatientInfoRouteBuilding['PLATAREA']

fig = px.box(PatientInfoRouteBuilding[PatientInfoRouteBuilding['vl_rat_average'] < 400], x="week", y="vl_rat_average")

mark_policy(fig)

fig.show()
PatientInfoRoute['BD_MGT_SN'] = PatientInfoRoute['building_code']



PatientInfoRouteBuilding = PatientInfoRoute.merge(building_info, on='BD_MGT_SN')

# PatientInfoRouteBuilding['건폐율'] = PatientInfoRouteBuilding['ARCHAREA'] / PatientInfoRouteBuilding['PLATAREA']

fig = px.box(PatientInfoRouteBuilding[PatientInfoRouteBuilding['TOTALAREA'] < 40000], x="week", y="TOTALAREA")

mark_policy(fig)

fig.show()
PatientInfoRoute['BD_MGT_SN'] = PatientInfoRoute['building_code']



PatientInfoRouteBuilding = PatientInfoRoute.merge(building_info, on='BD_MGT_SN')

PatientInfoRouteBuilding['Architecture area / Land area'] = PatientInfoRouteBuilding['ARCHAREA'] / PatientInfoRouteBuilding['PLATAREA']

fig = px.box(PatientInfoRouteBuilding[PatientInfoRouteBuilding['Architecture area / Land area'] < 1], x="week", y="Architecture area / Land area")

mark_policy(fig)

fig.show()