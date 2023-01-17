# 데이터 불러오기



import pandas as pd

import numpy as np

from collections import Counter



df_train = pd.read_json('../input/arena-res/train.json', encoding='utf-8') # 훈련 데이터

df_song = pd.read_json('../input/arena-res/song_meta.json', encoding='utf-8') # 곡 데이터
df_train
df_song
# Counter의 most_common() 반환. 

def most_common_data(row, column):

    song_list=row['songs']

    c= Counter()

    for song in song_list:

        try:

            data = df_song.iloc[song][column]

            c.update({data:1})

        except:

            if len(df_song.iloc[song][column]) <= 0:

                continue

            for d in df_song.iloc[song][column]:

                c.update({d:1})

    if len(c.most_common()) <= 0:

        return ['없음']

    return c.most_common()
def gnr_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return most_common_data(row, 'song_gn_gnr_basket')[0][0]
# Counter의 most_common() 반환. 

def gnr_name_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    data = most_common_data(row, 'song_gn_gnr_basket')[0][0]

    return gn_to_name(data)
# 백분율 계산 위해 전체 길이를 반환

def calc_per(row, column):

    c= Counter()

    count = 0

    song_list=row['songs']

    for song in song_list:

        count += 1

        try:

            data = df_song.iloc[song][column]

            c.update({data:1}) # 만약 단일 데이터가 아니라 리스트라면 에러가 난다.

        except:

            for d in df_song.iloc[song][column]:

                c.update({d:1})

    

    if len(c.most_common()) <= 0:

        return 0

    per = c.most_common()[0][1] / count * 100

    

    return per      

                    
def gnr_per_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return calc_per(row, 'song_gn_gnr_basket')
def artist_name_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return most_common_data(row, 'artist_name_basket')[0][0]
def artist_id_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return most_common_data(row, 'artist_id_basket')[0][0]
def artist_per_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return calc_per(row, 'artist_id_basket')
def album_name_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return most_common_data(row, 'album_name')[0][0]
def album_id_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return most_common_data(row, 'album_id')[0][0]
def album_per_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return calc_per(row, 'album_id')
def year_per_most_common_data(row):

    print(row[0], end='/')

    if row[0] % 10000 == 0:

        print(row[0])

    return sort_year(row)[0][1] / count_total_len(row) * 100
# 백분율 계산 위해 전체 길이를 반환

def count_total_len(row):

    song_list=row['songs']

    count = len(row['songs']) 

    return count
def sort_year(row):

    c= Counter()

    for i in most_common_data(row, 'issue_date'):

        year = int(str(i[0])[0:4])

        c.update({year:1})

    

    return c.most_common()



#sort_year(1)
def sort_year_name(row):

    c= Counter()

    for i in most_common_data(row, 'issue_date'):

        year = int(str(i[0])[0:4])

        c.update({year:1})

    

    return c.most_common()[0][0]
def sort_season(row):

    c= Counter()

    for i in most_common_data(row, 'issue_date'):

        #print(i)

        try:

            month = int(str(i[0])[5:6])

        except:

            continue

            

        season = ''

        if month >= 3 and month <= 5:

            season = "봄"

        elif month >= 6 and month <= 8:

            season = "여름"

        elif month >= 9 and month <= 11:

            season = "가을"

        else:

            season = '겨울'

            

        #print(season)

        

        c.update({season:1})

    

    return c.most_common()[0][0]



#sort_season(1)
def sort_season_per(row):

    c= Counter()

    for i in most_common_data(row, 'issue_date'):

        #print(i)

        try:

            month = int(str(i[0])[5:6])

        except:

            continue

            

        season = ''

        if month >= 3 and month <= 5:

            season = "봄"

        elif month >= 6 and month <= 8:

            season = "여름"

        elif month >= 9 and month <= 11:

            season = "가을"

        else:

            season = '겨울'

        c.update({season:1})

    

    

    return c.most_common()[0][1] / count_total_len(row) * 100

from datetime import datetime



# 날짜 표준편차 계산

def calc_std(row):

    li = []

    c= Counter()

    song_list=row['songs']

    for song in song_list:

        data = df_song.iloc[song]['issue_date']

        # 숫자로 저장된 날짜를 날짜형식으로 변환

        try:

            date2 = datetime.strptime(str(data), '%Y%m%d')

            #print(date2)



        # 날짜형식에 안맞는 데이터들 처리  

        except:

            #print(song)

            #print("곡 번호:",df_song.iloc[song]['id'])

            #print("날짜: ",data)

            if data < 1 or int(str(data)[-2:]) > 31:

                #print("continue로 생략")

                continue

            elif str(data)[-4:] == "0000":

                date2 = datetime.strptime(str(data+701), '%Y%m%d')

            elif str(data)[4:6] == "00":

                date2 = datetime.strptime(str(data+701), '%Y%m%d')

            elif str(data)[-2:] == "00":

                date2 = datetime.strptime(str(data+1), '%Y%m%d')

            else:

                continue





        zero = datetime.strptime('00010101', '%Y%m%d')

        #print(zero)

        date2 = (date2 - zero).days / 100

        #print(date2)

        li.append(date2)

        

    return np.std(li)

genre_gn_all = pd.read_json('../input/arena-res/genre_gn_all.json', typ = 'series', encoding='utf-8')



dict1 = dict(zip(genre_gn_all.index, genre_gn_all.values))

dict1
def gn_to_name(gn):

    

    try :

        return dict1[gn]

    except:

        return '없음'
# df_train['plylst_title'][0]
# from tqdm import tqdm



# def add_to_df(start, end, df):

#     for i in tqdm(range(start, end)):

#         df.loc[i, ['제목']] = df_train['plylst_title'][i]



#         df.loc[i, ['대분류_장르이름']] = gn_to_name(most_common_data(i, 'song_gn_gnr_basket')[0][0])

#         df.loc[i, ['대분류_장르코드']] = most_common_data(i, 'song_gn_gnr_basket')[0][0]

#         df.loc[i, ['대분류_장르_비율']] = calc_per(i, 'song_gn_gnr_basket')



# #         df.loc[i, ['상세분류_장르']] = most_common_data(i, 'song_gn_dtl_gnr_basket')[0][0]

# #         df.loc[i, ['상세분류_장르_비율']] = most_common_data(i, 'song_gn_dtl_gnr_basket')[0][1] / count_total_len(i, 'song_gn_dtl_gnr_basket') * 100

        

#         df.loc[i, ['아티스트_이름']] = most_common_data(i, 'artist_name_basket')[0][0]

#         df.loc[i, ['아티스트_아이디']] = most_common_data(i, 'artist_id_basket')[0][0]

#         df.loc[i, ['아티스트_아이디_비율']] = calc_per(i, 'artist_id_basket')

        

#         df.loc[i, ['앨범_이름']] = most_common_data(i, 'album_name')[0][0]

#         df.loc[i, ['앨범_아이디']] = most_common_data(i, 'album_id')[0][0]

#         df.loc[i, ['앨범_아이디_비율']] = calc_per(i, 'album_id')

        

#         #df.loc[i, ['발매일']] = most_common_data(i, 'issue_date')[0][0]

#         #df.loc[i, ['발매일_비율']] = most_common_data(i, 'issue_date')[0][1] / count_total_len(i, 'issue_date') * 100

#         df.loc[i, ['발매년도']] = sort_year(i)[0][0]

#         df.loc[i, ['발매년도_비율']] = sort_year(i)[0][1] / count_total_len(i, 'issue_date') * 100

        

#         df.loc[i, ['발매계절']] = sort_season(i)[0][0]

#         df.loc[i, ['발매계절_비율']] = sort_season(i)[0][1] / count_total_len(i, 'issue_date') * 100

        

#         df.loc[i, ['발매_표준편차']] = calc_std(i)
# # 저장할 빈 데이터프레임 생성

# test = pd.DataFrame()



# # 컬럼 생성을 위한 작업

# test['제목'] = None



# test['대분류_장르이름'] = None

# test['대분류_장르코드'] = None

# test['대분류_장르_비율'] = None



# # test['상세분류_장르'] = None

# # test['상세분류_장르_비율'] = None



# test['아티스트_이름'] = None

# test['아티스트_아이디'] = None

# test['아티스트_아이디_비율'] = None



# test['앨범_이름'] = None

# test['앨범_아이디'] = None

# test['앨범_아이디_비율'] = None



# #test['발매일'] = None

# #test['발매일_비율'] = None

# test['발매년도'] = None

# test['발매년도_비율'] = None

# test['발매계절'] = None

# test['발매계절_비율'] = None



# test['발매_표준편차'] = None
import pandas as pd

import numpy as np

from collections import Counter



test = pd.DataFrame()

test['id'] = df_train['id'][:10]

test['songs'] = df_train['songs'][:10]

test = test.sort_values(['id'], ascending=[True])





test['대분류_장르이름'] = None

test['대분류_장르이름'] = test.apply(gnr_name_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['대분류_장르코드'] = None

test['대분류_장르코드'] = test.apply(gnr_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['대분류_장르_비율'] = None

test['대분류_장르_비율'] = test.apply(gnr_per_most_common_data, axis=1)

print('\n',test, end='\n\n')





test['아티스트_이름'] = None

test['아티스트_이름'] = test.apply(artist_name_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['아티스트_아이디'] = None

test['아티스트_아이디'] = test.apply(artist_id_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['아티스트_아이디_비율'] = None

test['아티스트_아이디_비율'] = test.apply(artist_per_most_common_data, axis=1)

print('\n',test, end='\n\n')





test['앨범_이름'] = None

test['앨범_이름'] = test.apply(album_name_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['앨범_아이디'] = None

test['앨범_아이디'] = test.apply(album_id_most_common_data, axis=1)

print('\n',test, end='\n\n')



test['앨범_아이디_비율'] = None

test['앨범_아이디_비율'] = test.apply(album_per_most_common_data, axis=1)

print('\n',test, end='\n\n')





test['발매년도'] = None

test['발매년도'] = test.apply(sort_year_name, axis=1)

print('\n',test, end='\n\n')





test['발매년도_비율'] = None

test['발매년도_비율'] = test.apply(year_per_most_common_data, axis=1)

print('\n',test, end='\n\n')





test['발매계절'] = None

test['발매계절'] = test.apply(sort_season, axis=1)

print('\n',test, end='\n\n')



test['발매계절_비율'] = None

test['발매계절_비율'] = test.apply(sort_season_per, axis=1)

print('\n',test, end='\n\n')



test['발매_표준편차'] = None

test['발매_표준편차'] = test.apply(calc_std, axis=1)

print('\n',test, end='\n\n')







test
# # 데이터프레임에 추가



# add_to_df(0, 10, test)

# test
import distutils.dir_util



distutils.dir_util.mkpath("arena-res/eda/")

test.to_csv('arena-res/eda/eda_test.csv', index=False)



pd.read_csv('arena-res/eda/eda_test.csv')