import numpy as np

import pandas as pd

import matplotlib.pylab as plt

from math import isnan

import seaborn as sb

import copy

import datetime



def translate_df(df, key_list, JPtoEN_dict_list):



    df_en = copy.deepcopy(df)



    for key, JPtoEN_dict in zip(key_list, JPtoEN_dict_list):

        if key == 'column':

            df_en = df.rename(columns=JPtoEN_dict)



        else:

            df_en[key] = df_en[key].replace(JPtoEN_dict)



    return df_en



def clean_up_time_data(df):

    date_str_list = []

    day_str_list = []

    for ind, d in df.iterrows():

        a = datetime.datetime(d['year'], d['month'], d['day'])

        date_str_list.append(a.strftime("%Y/%m/%d"))

        day_str_list.append(a.strftime("%a"))

    



    df['date'] = date_str_list

    df['day_of_the_week'] = day_str_list 

    

    df = df.drop(columns=['index', 'year', 'month', 'day', 'prefecture'])

    

    return df





def get_test_df():

    # data source: https://toyokeizai.net/sp/visual/tko/covid19/en.html



    url2 = 'https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures-2.csv'

    test_df = pd.read_csv(url2)

    #test_df.keys()



    columns_dict = {'年': 'year', '月': 'month', '日': 'day', '都道府県': 'prefecture',

                    'PCR検査陽性者数': 'total_tested_positive_okinawa', 'PCR検査人数': 'total_tests', 'Unnamed': 'Unnamed'}

    test_df = test_df.rename(columns=columns_dict)



    test_oki_df = test_df[test_df.prefecture == '沖縄県'].reset_index()



    test_oki_df = clean_up_time_data(test_oki_df)

    test_oki_df['daily_tested_positive_okinawa'] = test_oki_df['total_tested_positive_okinawa'].diff()

    test_oki_df['daily_tests'] = test_oki_df['total_tests'].diff()

    

    test_oki_df = test_oki_df.drop(columns=['陽性率', '検査人数の集計期間に対応する陽性者数'])

    

    key_list = ['total_tests', 'total_tested_positive_okinawa', 'daily_tests', 'daily_tested_positive_okinawa']

    for key in key_list:

        test_oki_df[key] = test_oki_df[key].astype('float64')#pd.Int64Dtype())

    

    return test_oki_df

    





def get_status_df():

    url = 'https://raw.githubusercontent.com/kaz-ogiwara/covid19/master/data/prefectures.csv'

    status_df = pd.read_csv(url)



    columns_dict = {'年': 'year', '月': 'month', '日': 'day', '都道府県': 'prefecture',

                    '患者数（2020年3月28日からは感染者数）': 'patients', '現在は入院等': 'hospitalized', '退院者': 'discharged', '死亡者':'dead'}



    status_df_en = translate_df(status_df, ['column'], [columns_dict])

    status_oki_df = status_df_en[status_df_en.prefecture == '沖縄県'].reset_index()



    status_oki_df = clean_up_time_data(status_oki_df)

    stauts_oki_df = status_oki_df[['date', 'day_of_the_week', 'patients', 'hospitalized', 'discharged', 'dead']]

    

    key_list = ['patients', 'hospitalized', 'discharged', 'dead']

    for key in key_list:

        status_oki_df[key] = status_oki_df[key].astype('float64')#pd.Int64Dtype())

        

    status_oki_df['total_tested_positive_okinawa'] = status_oki_df['patients']



    return status_oki_df



def get_summary_df():

    status_df = get_status_df()

    test_df = get_test_df()





    url  = 'https://raw.githubusercontent.com/Code-for-OKINAWA/covid19/development/tool/downloads/status.csv'

    df = pd.read_csv(url)



    columns_dict = {'更新時間':'date', '検査人数累計':'total_tests', '検査実施人数':'daily_tests', '重症':'severe', '輸入病例':'total_tested_positive_imported', '県関係者陽性者数':'total_tested_positive_okinawa', '入院中':'hospitalized', '入院調整中':'to_be_hospitalized',

           '宿泊施設療養中':'resting_at_hotel', '自宅療養中':'resting_at_home', '入院勧告解除':'discharged', '死亡退院':'dead'}

    

    columns_dict_inv = {value:key for key, value in columns_dict.items()}

    col_EN = list(columns_dict.values())

    col_JP = list(columns_dict.keys())



    df= translate_df(df, ['column'], [columns_dict])





    date_str_list = []

    day_str_list = []

    for ind, d in df.iterrows():

        a = datetime.datetime.strptime(d['date'][:-6], "%Y/%m/%d")

        date_str_list.append(a.strftime("%Y/%m/%d"))

        day_str_list.append(a.strftime("%a"))



    df=df.drop(columns=['date'])

    df['date'] = date_str_list

    df['day_of_the_week'] = day_str_list 



    #summary_df = pd.merge(df, status_df, how='outer')

    status_test_df  = pd.merge(test_df, status_df, how='outer')

    summary_df =  pd.merge(status_test_df[:39], df, how='outer')#pd.concat([df, status_test_df])#, how='outer')

    summary_df = summary_df[col_EN]

    summary_df_JP = translate_df(summary_df, ['column'], [columns_dict_inv])

    

    return summary_df, summary_df_JP



def plot_summary(summary_df):

    key_list = list(summary_df.keys())



    num=len(key_list[1:])



    fig, axs = plt.subplots(num, 1, figsize=(10, 5*num), sharex=True)

    for ind, key in enumerate(key_list[1:]):

   

        sb.barplot(x='date',y=key, data=summary_df, ax=axs[ind])

        plt.xticks(rotation='vertical')



summary_df, summary_df_JP = get_summary_df()

summary_df.to_csv('status_en.csv')

summary_df_JP.to_csv('status.csv', encoding='utf-8-sig')

plot_summary(summary_df)
