# importing libraries onto kernel
!pip install geocoder==1.3 geopandas folium pandas numpy matplotlib seaborn
#!pip install ipywidgets
#!jupyter nbextension enable --py --sys-prefix widgetsnbextension
#%%writefile okinawa_covid19_map.py

# Last Updated: April 20, 2020
# Author: Soshi Mizutani (soshi.mizutani@oist.jp)

# 必要なライブラリのインストール
#!pip install geocoder==1.3 geopandas folium pandas numpy matplotlib seaborn

from folium.plugins import MarkerCluster, FeatureGroupSubGroup
from folium import LayerControl, Map, Marker, Icon, TileLayer
import geocoder
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from math import isnan
import seaborn as sb
import datetime
import copy
sb.set('poster')


def load_data():
    # load raw data (source: 'https://okinawa.stopcovid19.jp/')
    url = 'https://raw.githubusercontent.com/Code-for-OKINAWA/covid19/development/data/data.json'
    all_df = pd.read_json(url, orient='columns')

    df = pd.DataFrame(all_df['patients']['data'])

    df = df.drop(columns=['備考'])

    return df, all_df


def translate_df(df, key_list, JPtoEN_dict_list):

    df_en = copy.deepcopy(df)

    for key, JPtoEN_dict in zip(key_list, JPtoEN_dict_list):
        if key == 'column':
            df_en = df.rename(columns=JPtoEN_dict)

        else:
            df_en[key] = df_en[key].replace(JPtoEN_dict)

    return df_en


def load_data_en():

    # load raw data in JP
    df, all_df = load_data()

    # translate from JP to EN
    column_dict = {'date': 'date', '居住地': 'place', '年代': 'age',  '性別': 'gender',
                   '確定日': 'confirm_date', '状態': 'status', '退院': 'discharged'}
    age_dict = {'10代': '10s', '20代': '20s', '30代': '30s', '40代': '40s', '50代': '50s',
                '60代': '60s', '70代': '70s', '80代': '80s', '90代': '90s', '不明': 'N/A', '調査中': 'N/A', '確認中':'N/A'}
    gender_dict = {'女性': 'Female', '男性': 'Male', '不明': 'N/A', '調査中': 'N/A', '確認中':'N/A'}
    status_dict = {'退院': 'discharged', '入院勧告解除': 'discharged', '入院': 'hospitalized',
                   '調査中': 'N/A', '入院調整中': 'being arranged', '入院調整': 'to be hospitalized', '不明': 'N/A', '確認中':'N/A'}

    key_list = ['column', 'age', 'gender', 'status']
    JPtoEN_dict_list = [column_dict, age_dict, gender_dict, status_dict]
    df_en = translate_df(df, key_list, JPtoEN_dict_list)

    # shorten confirm-date
    for ind, d in df_en.iterrows():

        df_en['confirm_date'][ind] = df_en['confirm_date'][ind][5:10]

    return df_en


def get_count_df(data_df):
    count_df = data_df['place'].value_counts().rename_axis('place').reset_index(name='counts')
    return count_df


def get_loc_df():
    
    place_list = ['那覇市', '中部保健所管内', '南部保健所管内', '沖縄市', '浦添市', '豊見城市', '宜野湾市', '石垣市',
       '名護市', '南城市', 'うるま市', '糸満市', '北部保健所管内','宮古保健所管内', '宮古島市', '調査中', '県外']
    
    lat_list = [26.2122345, 26.319941 , 26.192634 , 26.3343738, 26.249754 ,
       26.1772381, 26.2815839, 24.366098 , 26.5914524, 26.142512 ,
       26.384705 , 26.106017 , 
                26.633257,
                24.790582,
                24.805460, 26.32, 26.5]
    
    lng_list = [127.6791452, 127.763825 , 127.729369 , 127.8056597, 127.716591 ,
       127.6863791, 127.7785754, 124.163499 , 127.9773062, 127.765058 ,
       127.851324 , 127.686066, 
           128.157012,
                125.298005,
                125.281102, 127.69, 127.69]

    loc_df = pd.DataFrame(
        {'place': place_list,
         'lat': lat_list,
         'lng': lng_list})

    return loc_df



def make_map(data_df, loc_df):
    OK_COORDINATES = (26.43, 127.8)

    # base map
    my_map = Map(location=OK_COORDINATES, zoom_start=10.12, min_zoom=8, max_zoom=13, tiles='cartodb positron')

    # tiles = [”OpenStreetMap”, ”Stamen Terrain”, “Stamen Toner”, “Stamen Watercolor”, ”CartoDB positron”, “CartoDB dark_matter”, ”Mapbox Bright”, “Mapbox Control Room” ]
    tile_options = ['cartodb positron', 'stamen terrain']
    tile_names = ["simple map", "terrain map"]
    for name, opt in zip(tile_names, tile_options):
        TileLayer(opt).add_to(my_map, name=name)

    # cluster object
    cluster = MarkerCluster(control=False)
    my_map.add_child(cluster)

    group_list = []
    for age_id in range(10):
        if age_id == 9:
            age_str = 'N/A'
        else:
            age_str = '%d0s' % (age_id + 1)

        group = FeatureGroupSubGroup(cluster, age_str)
        my_map.add_child(group)
        group_list.append(group)

    # add a marker for every record in the filtered data, use a clustered view
    for ind, d in data_df.iterrows():

        # make pop-up string
        place = d['place']
        age = d['age']
        gender = d['gender']
        conf_date = d['confirm_date']
        status = d['status']

        if age == 'N/A':
            age_id = 9
        else:
            age_id = int(age[0])-1
        
        pop_str = 'date:%s, age:%s' % (conf_date, age)
        

        if gender == 'Male':
            icon = Icon(color='blue', icon='male', prefix='fa')
        elif gender == 'Female':
            icon = Icon(color='red', icon='female', prefix='fa')
        else:
            icon = Icon(color='black', icon='question-circle', prefix='fa')

        # get coordinates
        try:
            lat_value = loc_df[loc_df.place == place].lat.values[0]
            lng_value = loc_df[loc_df.place == place].lng.values[0]
        except:
            lat_value = loc_df[loc_df.place == '県外'].lat.values[0]
            lng_value = loc_df[loc_df.place == '県外'].lng.values[0]
            pop_str = 'date:%s, age:%s, place=%s' % (conf_date, age, 'outside Okinawa')
            
        #print(place, lat_value, lng_value)

        # create marker with coordinates & pop-up info
        marker = Marker(location=[lat_value,  lng_value],
                        popup=pop_str, icon=icon)

        # add marker to gorup
        group_list[age_id].add_child(marker)

    LayerControl().add_to(my_map)

    my_map.save("covid19_okinawa_map.html")
    display(my_map)

    return my_map




# data
data_df = load_data_en()
loc_df = get_loc_df()

# plot map
my_map = make_map(data_df, loc_df)

# def get_coordinate(placename):
#     """get coordinate of placename """
#     ret = geocoder.osm(placename, timeout=50.0)

#     #print(placename, ret.latlng)
#     return ret

# def get_loc_df(data_df):
#    # get location's coordinate automatically

#     count_df = get_count_df(data_df)

#     lat_list = []
#     lng_list = []


#     for ind, address in enumerate(count_df.place):

#         if address[:2]=='南部':#保健所
#             place = '沖縄県島尻郡南風原町'#字宮平212'

#         elif address[:2]== '中部':#部保健所
#             place = '沖縄県北谷町役場'#美原1-6-28'
            
#         elif address[:2]=='北部':
#             place = '沖縄県名護市'

#         elif address[0]== 'う':#るま市
#             place = address

#         elif address[0]== '東':#京都
#             place = address
            
#         elif address[0]== '大':#坂
#             place = '大阪府大阪市'

#         else:
#             place = '沖縄県' + address

#         ret = get_coordinate(place)
#         print(place, ret.latlng)

#         if address=='不明':
#             lat, lng = 26.32, 127.69 # put it somewhere in the close ocean
#             lat_list.append(lat)
#             lng_list.append(lng)

#         else:
#             lat_list.append(ret.latlng[0])
#             lng_list.append(ret.latlng[1])


#     loc_df = pd.DataFrame(
#     {'place': count_df.place,
#      'counts': count_df.counts,
#      'lat': lat_list,
#     'lng': lng_list})
    
#     return loc_df


# def replace_place(data_df, oldplace, newplace):
#     data_df2 = copy.deepcopy(data_df)
#     data_df2=data_df2.replace(oldplace, newplace)
#     return data_df2
# #'北部保健所管内','宮古保健所管内'
# oldplace='糸満市'
# newplace='北部保健所管内'
# data_df2=replace_place(data_df, oldplace, newplace)
# my_map2=make_map(data_df2, loc_df)

sb.set_palette('colorblind')


def get_stat_df(data_df):
    case_df = data_df['confirm_date'].value_counts().rename_axis('confirm_date').reset_index(
        name='counts').sort_values('confirm_date', ascending=True)

    age_df = data_df['age'].value_counts().rename_axis('age').reset_index(name='counts')
    age_id = []
    for age in age_df.age:

        if age == 'N/A':
            age_id.append(10)
        else:
            age_id.append(int(age[0]))

    age_df['age_id'] = age_id
    age_df = age_df.sort_values('age_id')

    gender_df = data_df['gender'].value_counts().rename_axis('gender').reset_index(name='counts')
    status_df = data_df['status'].value_counts().rename_axis('status').reset_index(name='counts')

    return case_df, age_df, gender_df, status_df


def make_plot(data_df):

    case_df, age_df, gender_df, status_df = get_stat_df(data_df)
    # ===
    fig, ax1 = plt.subplots(figsize=(20, 15))

    ax1.plot(case_df['confirm_date'], case_df['counts'], '-o',  label='daily new cases', color='blue')
    ax1.plot(case_df['confirm_date'], case_df['counts'].cumsum(), '-o', label='total', color='red')
    plt.xticks(rotation='vertical')
    ax1.set_ylabel('counts')
    ax1.set_xlabel('date')
    __, all_df = load_data()
    last_update = all_df['lastUpdate']['date']
    titlename = 'COVID19 in Okinawa (last updated: %s)' % last_update
    ax1.set_title(titlename)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    count_now = case_df['counts'].cumsum().tail(1).values[0]
    count_now2 = case_df['counts'].tail(1).values[0]
    date_now = case_df['confirm_date'].tail(1).values[0]
    ax1.annotate(str(count_now), (date_now, count_now+0.3), color='red')
    ax1.annotate(str(count_now2), (date_now, count_now2+0.3), color='blue')
    ax1.grid(True)

    # unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.2, 0.65, 0.4, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    #ax2.pie(age_df['counts'].values, labels=age_df['age'].values, autopct='%1.1f%%', shadow=False, startangle=90, counterclock=False, pctdistance=0.8)
    sb.barplot(x="age", y="counts", data=age_df, ax=ax2)

    left2, bottom2, width2, height2 = [0.2, 0.4, 0.15, 0.15]
    ax3 = fig.add_axes([left2, bottom2, width2, height2])
    sb.barplot(x="gender", y="counts", data=gender_df, palette=['blue', 'red'], ax=ax3)

#     left3, bottom3, width3, height3 = [0.4, 0.4, 0.15, 0.15]
#     ax3 = fig.add_axes([left3, bottom3, width3, height3])
#     sb.barplot(x="status", y="counts", data=status_df, ax=ax3)
#     plt.xticks(rotation='vertical')

    plt.savefig('covid19_okinawa_counts.png')



def make_test_plot():
    # data source: https://toyokeizai.net/sp/visual/tko/covid19/en.html

    url2 = 'https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures-2.csv'
    test_df = pd.read_csv(url2)
    test_df.keys()

    columns_dict = {'年': 'year', '月': 'month', '日': 'day', '都道府県': 'prefecture',
                    'PCR検査陽性者数': 'test-positive', 'PCR検査人数': 'test-total', 'Unnamed': 'Unnamed'}
    test_df = test_df.rename(columns=columns_dict)

    test_oki_df = test_df[test_df.prefecture == '沖縄県'].reset_index()

    date_str_list = []
    for ind, d in test_oki_df.iterrows():
        a = datetime.datetime(d['year'], d['month'], d['day'])
        date_str_list.append(a.strftime("%m-%d-%a"))

    test_oki_df['date'] = date_str_list
    test_oki_df['daily-test-positive'] = test_oki_df['test-positive'].diff()
    test_oki_df['daily-test-total'] = test_oki_df['test-total'].diff()

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    sb.barplot(x='date', y='daily-test-total', data=test_oki_df,
               label='total', color='grey', ax=ax1)
    sb.barplot(x='date', y='daily-test-positive', data=test_oki_df,
               label='positive', color='red', ax=ax1)
    plt.legend()
    plt.ylabel('test counts')
    plt.xticks(rotation='vertical')
    plt.title('PCR tests in Okinawa')
    plt.savefig('covid19_okinawa_test.png')



def get_japan_stat_df():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/demography.csv').fillna(int(0))
    df = df.rename(columns={'年代': 'age', '死亡': 'dead', '重症': 'severe',
                            'その他': 'others', '有症者': 'sick', '無症者': 'ok', '症状確認中': 'unknown'})
    df['counts'] = df.iloc[:, 1:].sum(axis=1)
    age_dict = {'10歳未満': '0s', '10代': '10s', '20代': '20s', '30代': '30s', '40代': '40s', '50代': '50s',
                '60代': '60s', '70代': '70s', '80代': '80s', '80代以上': '80s', '90代以上': '90s', '不明': 'N/A'}
    japan_df = translate_df(df, ['age'], [age_dict])

    return japan_df


def oki_japan_plot(data_df, japan_df):

    case_df, age_df, gender_df, status_df = get_stat_df(data_df)
    age_df = age_df.append({'age': '0s', 'counts': 0, 'age_id': 0}, ignore_index=True)
    age_df = age_df.append({'age': '90s', 'counts': 0, 'age_id': 9}, ignore_index=True)
    age_df = age_df.sort_values('age_id').reset_index()
    # ===
    fig, axs = plt.subplots(2, 1, figsize=(20, 15), sharex=True)
    ax1, ax2 = axs.flatten()

    sb.set_palette("hls", 8)
    p1 = sb.barplot(x="age", y="counts", data=age_df, ax=ax1)
    plt.gca().set_prop_cycle(None)
    p2 = sb.barplot(x='age', y='counts', data=japan_df, ax=ax2)

    for index, row in japan_df.iterrows():
        p2.text(row.name, 20, int(row.dead), color='black', ha="center")

    ax1.set_title('Okinawa')
    ax2.set_title('all Japan')
    ax2.text(10.5, 1,  '(No. of deaths)', style='italic')

    plt.tight_layout()
    plt.savefig('covid19_okinawa_japan_age.png')


def del_day(day0, df):
    df2 = copy.deepcopy(df)
    delta_day_list = []
    for ind, d in df.iterrows():
        d1 = datetime.date(day0[0], day0[1], day0[2])
        d2 = datetime.date(d.year, d.month, d.day)
        delta_day = d2 - d1
        delta_day_list.append(delta_day.days)

    df2['delta_day'] = delta_day_list

    return df2


def plot_oki_fuku():
    # compare okinawa & fukuoka (fukuoka data scaled by pulation)
    df = pd.read_csv('https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures.csv')
    columns_dict = {'年':'year', '月':'month', '日':'day', '都道府県':'prefecture', '患者数（2020年3月28日からは感染者数）':'patient', '現在は入院等':'hospitalized', '退院者':'discharged', '死亡者':'death'}
    df = df.rename(columns=columns_dict)
    date_str_list=[]
    for ind, d in df.iterrows():
        a = datetime.datetime(d['year'], d['month'], d['day'])
        date_str_list.append(a.strftime("%m-%d"))

    df['date'] = date_str_list

    oki_df = df[df.prefecture=='沖縄県'].reset_index()
    fuku_df = df[df.prefecture=='福岡県'].reset_index()

    N_oki = 1.457#million
    N_fukuoka = 5.108 #million
    scale  = N_oki/N_fukuoka

    fuku_df2 = fuku_df
    fuku_df2['patient'] = fuku_df['patient']*scale

    N_ma = 3
    oki_df['patient_ma'] = oki_df['patient'].rolling(N_ma, center=True).mean() 
    fuku_df2['patient_ma'] = fuku_df2['patient'].rolling(N_ma, center=True).mean()
    
    fig, ax1=plt.subplots(1,1,figsize=(15,8))
    sb.scatterplot(x='date', y='patient', data=oki_df, label='Okinawa', color='red', ax=ax1)
    sb.lineplot(x='date', y='patient_ma', data=oki_df, label='Okinawa (3day-average)', color='red', ax=ax1, alpha=0.2) 
    sb.scatterplot(x='date', y='patient', data=fuku_df2, label='Fukuoka*', color='blue', ax=ax1, alpha=0.5)
    sb.lineplot(x='date', y='patient_ma', data=fuku_df2, label='Fukuoka* (3day-average)', color='blue', ax=ax1, alpha=0.2) 
    #ax1.set_yscale('log', basey=2)
    plt.legend()
    plt.ylabel('cases')
    plt.xticks(rotation='vertical')
    plt.title('No. of Covid19 cases in Okinawa and Fukuoka')
    plt.savefig('covid19_okinawa_fukuoka.png')
    
    
    # data since cases>20
#     oki_day0 = (2020, 4, 7)
#     fuku_day0 = (2020, 4, 5)

    # day since cases>10
    oki_day0 = (2020, 4, 5)
    fuku_day0 = (2020, 3, 31)
    oki_df = del_day(oki_day0, oki_df)
    fuku_df2 = del_day(fuku_day0, fuku_df2)
    
    t_list = np.arange(15)
    doubled_every_day= [2**(t+3) for t in t_list]
    doubled_every_2day = [2**(t/2+3) for t in t_list]
    doubled_every_7day = [2**(t/7+3) for t in t_list]
    
    fig, ax1=plt.subplots(1,1,figsize=(15,8))
    sb.scatterplot(x='delta_day', y='patient', data=oki_df[oki_df['delta_day']>-1], label='Okinawa', color='red', ax=ax1)
    sb.lineplot(x='delta_day', y='patient_ma', data=oki_df[oki_df['delta_day']>-1], label='Okinawa (3day-average)', color='red', ax=ax1, alpha=0.2) 
    sb.scatterplot(x='delta_day', y='patient', data=fuku_df2[fuku_df2['delta_day']>-1], label='Fukuoka*', color='blue', ax=ax1, alpha=0.5)
    sb.lineplot(x='delta_day', y='patient_ma', data=fuku_df2[fuku_df2['delta_day']>-1], label='Fukuoka* (3day-average)', color='blue', ax=ax1, alpha=0.2) 
    
    ax1.plot(t_list, doubled_every_day, color='grey', alpha=0.5)
    ax1.plot(t_list, doubled_every_2day, color='grey', alpha=0.5)
    ax1.plot(t_list, doubled_every_7day, color='grey', alpha=0.5)
    
    ax1.annotate('doubled every day', (10, 2**14), color='grey')
    ax1.annotate('every 2 day', (10, 2**10), color='grey')
    ax1.annotate('every week', (10, 2**4), color='grey')
    
    ax1.set_yscale('log', basey=2)
    ax1.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('cases')
    plt.xlabel('days since 10 cases')
    plt.xticks(np.arange(0, 21, 2), rotation='vertical')
    plt.title('No. of Covid19 cases in Okinawa and Fukuoka')
    plt.savefig('covid19_okinawa_fukuoka_log2.png')
    

def make_regional_plot(data_df):
#     place_list = ['那覇市', '中部保健所管内', '南部保健所管内', '沖縄市', '浦添市', '豊見城市', '宜野湾市', '石垣市',
#        '名護市', '南城市', 'うるま市', '糸満市', '北部保健所管内','宮古保健所管内', '宮古島市', '調査中', '県外']
    place_dict = {'南部保健所管内': 'Nanbu health center', '那覇市': 'Naha', '豊見城市': 'Tomishiro', '中部保健所管内': 'Chubu health center', '東京都': 'Tokyo', '宜野湾市': 'Ginowan', '浦添市': 'Urasoe', '南城市': 'Nanjo',
                  '沖縄市': 'Okinawa', '糸満市': 'Itoman', 'うるま市': 'Uruma', '名護市': 'Nago', '石垣市': 'Ishigaki', '調査中': 'N/A', '確認中':'N/A'}

    data_df = translate_df(data_df, ['place'], [place_dict])
    count_df = get_count_df(data_df)
    place_list = count_df.place[:5]
    date_list = data_df.confirm_date.unique()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    for ind, place in enumerate(place_list):
        if place == 'N/A':
            pass
        else:
            data = data_df[data_df.place == place]['confirm_date'].value_counts(
            ).rename_axis('confirm_date').reset_index(name='counts')
            c_list = []

            for day in date_list:
                try:
                    c = data[data.confirm_date == day].counts.values[0]
                except:
                    c = 0

                c_list.append(c)

            #count_data = data_df[data_df.place == place]['confirm_date'].value_counts().rename_axis('confirm_date').reset_index(name='counts').sort_values('confirm_date', ascending=True).reset_index()
            labelname = '%d:%s' % (ind+1, place)
            if place == 'Okinawa':
                alpha = 1
            elif ind < 5:
                alpha = 0.6
            else:
                alpha = 0.2
            ax.plot(date_list, np.cumsum(c_list), '-o', label=labelname, alpha=alpha)

    plt.xticks(rotation='vertical')
    ax.set_ylabel('total cases')
    ax.set_title('Covid19 Okinawa, regional ranking top5')
    ax.legend()
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
    plt.savefig('covid19_okinawa_region.png')
    plt.show()
    
def make_age_plot(data_df):
#     place_list = ['那覇市', '中部保健所管内', '南部保健所管内', '沖縄市', '浦添市', '豊見城市', '宜野湾市', '石垣市',
#        '名護市', '南城市', 'うるま市', '糸満市', '北部保健所管内','宮古保健所管内', '宮古島市', '調査中', '県外']

    case_df, age_df, gender_df, status_df = get_stat_df(data_df)

    age_df = age_df.sort_values('counts', ascending=False)
    age_list = age_df.age[:5]
    date_list = data_df.confirm_date.unique()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))

    for ind, age in enumerate(age_list):
        if age == 'N/A':
            pass
        else:
            data = data_df[data_df.age == age]['confirm_date'].value_counts(
            ).rename_axis('confirm_date').reset_index(name='counts')
            c_list = []

            for day in date_list:
                try:
                    c = data[data.confirm_date == day].counts.values[0]
                except:
                    c = 0

                c_list.append(c)

            #count_data = data_df[data_df.place == place]['confirm_date'].value_counts().rename_axis('confirm_date').reset_index(name='counts').sort_values('confirm_date', ascending=True).reset_index()
            labelname = '%d:%s' % (ind+1, age)
#             if place == 'Okinawa':
#                 alpha = 1
#             elif ind < 5:
#                 alpha = 0.6
#             else:
#                 alpha = 0.2
            ax.plot(date_list, np.cumsum(c_list), '-o', label=labelname)

    plt.xticks(rotation='vertical')
    ax.set_ylabel('total cases')
    ax.set_title('Covid19 Okinawa, age ranking')
    ax.legend()
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
    plt.savefig('covid19_okinawa_age.png')
    plt.show()
    


def plot_mobility(city_name):

    df = pd.read_csv(
        'https://raw.githubusercontent.com/datasciencecampus/google-mobility-reports-data/master/csvs/international_local_area_trends_G20_20200410.csv')
    oki_df = df[df.location == city_name]
    category_key_list = oki_df.category.unique()

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))  # , sharex=True, sharey=True)
    for ind, key in enumerate(category_key_list):
        data = oki_df[oki_df.category == key].iloc[:, 3:]
        date_list = data.keys().values
        value_list = data.values.flatten()

        ax.plot(date_list, value_list, label=key)

    plt.legend()
    ax.set(xlabel="Date",
           ylabel="%",
           title="%s Mobility Report by Google" % city_name)
    plt.xticks(rotation='vertical')
    plt.show()
    
# basic stat
make_plot(data_df)

# regional ranking
make_regional_plot(data_df)

# age ranking
make_age_plot(data_df)

# PCR test
make_test_plot()

# compare okinawa & japan
japan_df = get_japan_stat_df()
oki_japan_plot(data_df, japan_df)

# compare okinawa & fukuoka
plot_oki_fuku()



# https://github.com/kaz-ogiwara/covid19/blob/master/data/prefectures.csv
# https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures.csv



def plot_oki_city(city_list, city_list_en, N_list):
    
    #load date
    df = pd.read_csv('https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures.csv')
    columns_dict = {'年':'year', '月':'month', '日':'day', '都道府県':'prefecture', '患者数（2020年3月28日からは感染者数）':'patient', '現在は入院等':'hospitalized', '退院者':'discharged', '死亡者':'death'}
    df = df.rename(columns=columns_dict)
    date_str_list=[]
    for ind, d in df.iterrows():
        a = datetime.datetime(d['year'], d['month'], d['day'])
        date_str_list.append(a.strftime("%m-%d"))

    df['date'] = date_str_list
    
    #==== city df=======
    N_ma = 3
    N_oki = 1.457#N_list[0]#million
    city_df_list = []
    for N, city in zip(N_list, city_list):
        scale  = N_oki/N
        city_df = df[df.prefecture==city].reset_index()
        city_df['patient'] = city_df['patient']*scale
        city_df['patient_ma'] = city_df['patient'].rolling(N_ma, center=True).mean()
        city_df_list.append(city_df)
        
    #====add NYC======
    us_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', delimiter=',')
    ny_df = us_df[us_df.Province_State=='New York'][us_df.Admin2=='New York'].iloc[:, 10:]
    date_list = ny_df.keys()[1:].values
    count_list = ny_df.values.flatten()[1:].astype(int)
    ny_dict = {'date':date_list, 'patient':count_list}
    ny_df = pd.DataFrame(ny_dict)
    N_ny = 8.399                                                                               
    scale = N_oki/N_ny                                                                            
    ny_df['patient'] = ny_df['patient']*scale
    ny_df['patient_ma'] = ny_df['patient'].rolling(N_ma, center=True).mean()
    city_df_list.append(ny_df)
    city_list_en.append('New York City')                                                                               
  
    #===linear plot =======
    fig, ax1=plt.subplots(1,1,figsize=(15,8))
    for city, city_df in zip(city_list_en[:-1], city_df_list[:-1]):
        alpha=1 if city=='Okinawa' else 0.5
            
        sb.scatterplot(x='date', y='patient', data=city_df, label=city,  ax=ax1, alpha=alpha)
        sb.lineplot(x='date', y='patient_ma', data=city_df, ax=ax1, alpha=0.2) 

    plt.legend()
    plt.ylabel('cases')
    plt.xticks(rotation='vertical')
    plt.title('No. of Covid19 cases in Okinawa and Fukuoka')
    plt.savefig('covid19_okinawa_city.png')
    
    #====log2 plot========
    
    # guided lines
    m=20
    l=np.log2(m)
    t_list = np.arange(15)
    doubled_every_day= [2**(t+l) for t in t_list]
    doubled_every_2day = [2**(t/2+l) for t in t_list]
    doubled_every_7day = [2**(t/7+l) for t in t_list]
    
    fig, ax1=plt.subplots(1,1,figsize=(15,8))
    
    city_df2_list=[]
    for city, city_df in zip(city_list_en, city_df_list):
        alpha = 1 if city=='Okinawa' else 0.5
        city_df = city_df.loc[city_df.patient> m].reset_index().copy()
        city_df['delta_day'] = copy.deepcopy(city_df.index.values)
        sb.lineplot(x='delta_day', y='patient_ma', data=city_df, label=city, ax=ax1, alpha=alpha) 
        city_df2_list.append(city_df)
        
    ax1.plot(t_list, doubled_every_day, color='grey', alpha=0.5)
    ax1.plot(t_list, doubled_every_2day, color='grey', alpha=0.5)
    ax1.plot(t_list, doubled_every_7day, color='grey', alpha=0.5)
    
    ax1.annotate('doubled every day', (10, 2**14), color='grey')
    ax1.annotate('every 2 day', (10, 2**10), color='grey')
    ax1.annotate('every week', (10, 2**5), color='grey')
    
    ax1.set_yscale('log', basey=2)
    ax1.legend()#loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([m, 2**15])
    plt.xlim([0,40])
    plt.ylabel('cases')
    plt.xlabel('days since %d cases' % m)
    plt.xticks(np.arange(0, 40, 2), rotation='vertical')
    plt.title('No. of Covid19 cases in Okinawa and other cities')
    plt.savefig('covid19_okinawa_city_log2.png')
    
    return city_df2_list



city_list = ['沖縄県', '東京都', '大阪府', '福岡県']#, '北海道']
city_list_en = ['Okinawa', 'Tokyo', 'Osaka', 'Fukuoka']#, 'Hokkaido']
N_list = [1.457, 9.273, 8.823,  5.108]#, 5.281]
city_df_list = plot_oki_city(city_list, city_list_en, N_list)
#!pip install -r ../input/requirements.txt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def SIR_model(y, t, N, beta, gamma):
    """
    The SIR model differential equations.
    y = (S, I, R), 
    where
    S = Susceptible
    I = Infectious
    R = Recovered
    
    N = total population
    beta = contact rate (per day)
    gamma = mean recovery rate (per day)
    (r0 = beta/gamma = basic reproduction rate)
    """
    S, I, R = y
    del_S = -beta * S * I / N
    del_I = beta * S * I / N - gamma * I
    del_R = gamma * I
    return del_S, del_I, del_R


def solve_SIR_model(N, N_day, T_recover, r0, I0, R0):
    
    # time grid = [0, N_day]
    t = np.arange(N_day)
    
    # set initial conditions vector y0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    # set gamma from basic reproduction rate: r0 = beta/gamma
    #gamma = beta / r0
    gamma = 1/T_recover
    beta = gamma * r0
    
    # solve SIR model by integrating it with initial condition
    ret = sp.integrate.odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S_t, I_t, R_t = ret.T # ret.shape = (N_day, 3)

    return t, S_t, I_t, R_t
    
def plot_SIR(t, S_t, I_t, R_t, T_recover, r0, show_all = True):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    if show_all:
        ax1.plot(t, S_t, label='Susceptible', color='blue', alpha=0.2)
        ax1.plot(t, R_t, label='Recoverd', color='green', alpha=0.2)
    else:
        pass
    ax1.plot(t, I_t, label='Infected', color='red')
   
    ax1.legend()
    ax1.set_xlabel('Day')
    gamma = 1/T_recover
    beta = gamma*r0
    titlename = '$T_{recover}$ = %g, $R_0$ = %g \n ($\\beta$ = %g, $\\gamma$ = %g)' % (T_recover, r0, beta, gamma)
    ax1.set_title(titlename)
    plt.show()
    
    
def simulate_SIR(N, N_day, T_recover, r0, I0, R0, show_all=True):
    t, S_t, I_t, R_t = solve_SIR_model(N, N_day, T_recover, r0, I0, R0)
    plot_SIR(t, S_t, I_t, R_t, T_recover, r0, show_all)


    
# SIR example
N = 1000
interact(simulate_SIR, N=(100, 1000, 100), N_day=fixed(92), T_recover=(5, 20, 1), r0=(0, 20, 1), I0=fixed(1), R0=fixed(0), show_all=True)

# N_oki = 1.457 * 10**6
# interact(simulate_SIR, N=fixed(N_oki), N_day=fixed(90), T_recover=(7, 30, 1), r0=(1, 30, 1), I0=fixed(1), R0=fixed(0), show_all=False)

