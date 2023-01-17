# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading every country's data

ca = pd.read_csv('../input/youtube-new/CAvideos.csv')

de = pd.read_csv('../input/youtube-new/DEvideos.csv')

fr = pd.read_csv('../input/youtube-new/FRvideos.csv')

gb = pd.read_csv('../input/youtube-new/GBvideos.csv')

ind = pd.read_csv('../input/youtube-new/INvideos.csv')

jp = pd.read_csv('../input/youtube-new/JPvideos.csv', engine='python')

kr = pd.read_csv('../input/youtube-new/KRvideos.csv', engine='python')

mx = pd.read_csv('../input/youtube-new/MXvideos.csv', engine='python')

ru = pd.read_csv('../input/youtube-new/RUvideos.csv', engine='python')

us = pd.read_csv('../input/youtube-new/USvideos.csv')
from langdetect import detect

url = re.compile(r"http(s)?:\/\/(\S)+")    # Descriptions often contain URL, so we will remove them, as they can disturb langdetect



def descr_languages(df):

    res = {}

    for idx, d in df.loc[:,['title', 'description']].iterrows():

        s = re.sub(url, '', str(d['title'])+' '+str(d['description']))

        try:

            lang = detect(s)

            if lang in res.keys():

                res[lang] = res[lang] + 1

            else:

                res[lang] = 1

        except:

            pass

    return res

print('Loading languages for Canada')

ca_lang = descr_languages(ca)

print(ca_lang)
print('Loading languages for Germany')

de_lang = descr_languages(de)

print(de_lang)

print('Loading languages for France')

fr_lang = descr_languages(fr)

print(fr_lang)

print('Loading languages for Great-Britain')

gb_lang = descr_languages(gb)

print(gb_lang)

print('Loading languages for India')

ind_lang = descr_languages(ind)

print(ind_lang)

print('Loading languages for Japan')

jp_lang = descr_languages(jp)

print(jp_lang)

print('Loading languages for South Korea')

kr_lang = descr_languages(kr)

print(kr_lang)

print('Loading languages for Mexico')

mx_lang = descr_languages(mx)

print(mx_lang)

print('Loading languages for Russia')

ru_lang = descr_languages(ru)

print(ru_lang)

print('Loading languages for USA')

us_lang = descr_languages(us)

print(us_lang)
def top3(lang):

    return sorted(lang.items(), key=lambda t: t[1], reverse=True)[:3]



print("CANADA  : "+str(top3(ca_lang)))

print("GERMANY : "+str(top3(de_lang)))

print("FRANCE  : "+str(top3(fr_lang)))

print("GB      : "+str(top3(gb_lang)))

print("INDIA   : "+str(top3(ind_lang)))

print("JAPAN   : "+str(top3(jp_lang)))

print("KOREA   : "+str(top3(kr_lang)))

print("MEXICO  : "+str(top3(mx_lang)))

print("RUSSIA  : "+str(top3(ru_lang)))

print("USA     : "+str(top3(us_lang)))
et_kr = []

for i, u in kr.iterrows():

    d = u['description']

    try:

        re.sub(url, '', d)

        lang = detect(d)

        if lang == 'et':

            et_kr.append(u)

    except:

        pass
print(et_kr[0]['description'])

print(et_kr[1]['description'])

print(et_kr[2]['description'])
def lang_pie(xx_lang):

    xx = sorted(xx_lang.items(), key=lambda t: t[1], reverse=True)

    res = xx[:5]

    c = 0

    for x  in xx[6:]:

        c += x[1]

    res.append(('others',c))

    return res

    

ca_lang_pie = lang_pie(ca_lang)

de_lang_pie = lang_pie(de_lang)

fr_lang_pie = lang_pie(fr_lang)

gb_lang_pie = lang_pie(gb_lang)

ind_lang_pie = lang_pie(ind_lang)

jp_lang_pie = lang_pie(jp_lang)

kr_lang_pie = lang_pie(kr_lang)

mx_lang_pie = lang_pie(mx_lang)

ru_lang_pie = lang_pie(ru_lang)

us_lang_pie = lang_pie(us_lang)



def pie(xx_lang_pie, title):

    size = []

    labels = []

    for x in xx_lang_pie:

        size.append(x[1])

        labels.append(x[0])

    fig, axs = plt.subplots()

    axs.set_title(title)

    axs.pie(size, labels=labels)



pie(ca_lang_pie,'CANADA')

pie(de_lang_pie,'GERMANY')

pie(fr_lang_pie,'FRANCE')

pie(gb_lang_pie,'GREAT BRITAIN')

pie(ind_lang_pie,'INDIA')

pie(jp_lang_pie,'JAPAN')

pie(kr_lang_pie,'SOUTH KOREA')

pie(mx_lang_pie,'MEXICO')

pie(ru_lang_pie,'RUSSIA')

pie(us_lang_pie,'USA')

    

#fig, axs = plt.subplots(4, 2)

#axs[0, 0].pie(ca_lang_pie[1], labels=ca_lang_pie[0])

def add_vids(videos, df, name):

    for idx, row in df.iterrows():

        if row['video_id'] is not None:

            if row['video_id'] in videos.keys():

                if name not in videos[row['video_id']]:              # Seems that a same viedo_id can appear various time for a same country...

                    videos[row['video_id']] = videos.get(row['video_id']) + [name]

            else:

                videos[row['video_id']] = [name]
videos = {}

print('Checking videos for Canada')

add_vids(videos, ca, 'ca')

print('Checking videos for Germany')

add_vids(videos, de, 'de')

print('Checking videos for France')

add_vids(videos, fr, 'fr')

print('Checking videos for Great-Britain')

add_vids(videos, gb, 'gb')

print('Checking videos for India')

add_vids(videos, ind, 'ind')

print('Checking videos for Japan')

add_vids(videos, jp, 'jp')

print('Checking videos for South Korea')

add_vids(videos, kr, 'kr')

print('Checking videos for Mexico')

add_vids(videos, mx, 'mx')

print('Checking videos for Russia')

add_vids(videos, ru, 'ru')

print('Checking videos for USA')

add_vids(videos, us, 'us')

print('Check completed')
spread_countries = {}

spread_number = {}

count = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

for k in videos.keys():

    count[len(videos[k])] = count.get(len(videos[k])) + 1

    if len(videos[k]) > 1:

        spread_countries[k] = videos[k]

        spread_number[k] = len(videos[k])

        

print(count)
spread_number_10 = []

for sn in spread_number.keys():

    if spread_number[sn] == 10:

        spread_number_10.append(sn)



for idx, row in jp.iterrows():

    if row['video_id'] in spread_number_10:

        print(row['title'])