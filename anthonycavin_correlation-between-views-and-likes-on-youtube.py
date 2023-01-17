import numpy as np
import pandas as pd
import seaborn as sns
col_list = ['video_id', 'views', 'likes']

us_yt = pd.read_csv('../input/youtube-new/USvideos.csv', usecols=col_list) #USA
ca_yt = pd.read_csv('../input/youtube-new/CAvideos.csv', usecols=col_list) #Canada
de_yt = pd.read_csv('../input/youtube-new/DEvideos.csv', usecols=col_list) #Germany
fr_yt = pd.read_csv('../input/youtube-new/FRvideos.csv', usecols=col_list) #France
gb_yt = pd.read_csv('../input/youtube-new/GBvideos.csv', usecols=col_list) #United Kingdom (Great Brittain)
in_yt = pd.read_csv('../input/youtube-new/INvideos.csv', usecols=col_list) #India
jp_yt = pd.read_csv('../input/youtube-new/JPvideos.csv', usecols=col_list) #Japan
kr_yt = pd.read_csv('../input/youtube-new/KRvideos.csv', usecols=col_list) #South Korea
mx_yt = pd.read_csv('../input/youtube-new/MXvideos.csv', usecols=col_list) #Mexico
ru_yt = pd.read_csv('../input/youtube-new/RUvideos.csv', usecols=col_list) #Russia

data_yt = pd.concat([us_yt, ca_yt, de_yt, fr_yt, gb_yt, in_yt, jp_yt, kr_yt, mx_yt, ru_yt], 
                    keys=['us', 'ca', 'de', 'fr', 'gb', 'in', 'jp', 'kr', 'mx', 'ru'])
data_yt
sns.regplot(x="likes", y="views", data=data_yt[['likes', "views"]])
data_yt[['likes', "views"]].corr()