import pandas as pd

import numpy as np



df_song = pd.read_json('../input/arena-res/song_meta.json') # 곡 데이터



df = df_song[['id', 'issue_date']]



from datetime import datetime



def calc_date(row):

    if row[0] % 10000 == 0:

        print(row[0])

    if row['issue_date'] == 0:

        return 0

    else:

        data = row['issue_date']

        date = ''

        try:

            date = datetime.strptime(str(data), '%Y%m%d') 

        except:

            if str(data)[-2:] > "31":

                return 0

            elif str(data)[-4:] == "0000":

                date = datetime.strptime(str(data+701), '%Y%m%d')

            elif str(data)[4:6] == "00" :

                date = datetime.strptime(str(data+700), '%Y%m%d')

            elif str(data)[-2:] == "00" :

                date = datetime.strptime(str(data+1), '%Y%m%d')

            else:

                return 0

            

        zero = datetime.strptime('19000101', '%Y%m%d') # 정상적으로 입력된 가장 빠른 날짜

            

        date = (date - zero).days

        return date

    

df['issue_date'] = df.apply(calc_date, axis=1)

df['issue_date'].astype(np.float32)



df

df.to_csv('issue_date_csv.csv', index=False)
import pandas as pd

df_train = pd.read_json('../input/arena-res/train.json') # 곡 데이터

df_train
df2 = df_train[['id', 'songs']]

df2['평균'] = 0

df2['표준편차'] = 0

df2 = df2.sort_values(["id"], ascending=[True])

df2
from datetime import datetime

import numpy as np



# 날짜 표준편차 계산

def calc_std(row):    

    if row[0] % 10000 == 0:

        print(row[0])

    li = []

    song_list=row['songs']

    

    for song in song_list:

        data = df.iloc[song]['issue_date']

        #print(data)

        if data <= 0:

            continue

        else:

            li.append(data)

    

    #print(np.std(li))

    return np.std(li)
from datetime import datetime

import numpy as np



# 날짜 평균 계산

def calc_mean(row):    

    if row[0] % 10000 == 0:

        print(row[0])

    li = []

    song_list=row['songs']

    

    for song in song_list:

        data = df.iloc[song]['issue_date']

        #print(data)

        if data <= 0:

            continue

        else:

            li.append(data)

    

#     print(np.std(li))

    return np.mean(li)
df2['표준편차'] = df2.apply(calc_std, axis=1)

df2
df2['평균'] = df2.apply(calc_mean, axis=1)

df2
df2.to_csv('issue_date_playlist.csv', index=False)