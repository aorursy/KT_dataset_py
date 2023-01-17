### packages for data manipilation ###

import numpy as np 

import pandas as pd 



### packages for plotting ###

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import iplot, init_notebook_mode



### web info extract###

from bs4 import BeautifulSoup

import requests



### other helpers ###

import os, json, gc, datetime

from datetime import timedelta



# just in case that Chinese character cannot be recognised

from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname="/kaggle/input/chinesecharacter/NotoSansHans-Regular.otf",size=15)



# import datasets

record = pd.read_csv('/kaggle/input/bangumi/record_2020_03_10.tsv',delimiter='\t')

user   = pd.read_csv('/kaggle/input/bangumi/user_2020_03_10.tsv'  ,delimiter='\t')
user.head(3)
record.loc[record.comment.notnull()].head(5)
record.loc[record.iid == 876].state.value_counts()
mirror_bgm = 'http://mirror.bgm.rin.cat/'

def rating_status(iid):

    """get rating status from the subject 

    return: a dictionary -- {rate: no. of rating}"""

    temp = requests.get(mirror_bgm + f'subject/{iid}')

    temp.encoding = "UTF-8"

    temp = temp.text

    bans = BeautifulSoup(temp, 'html.parser') 

    

    ouput = [eval(i.get_text()[1:-1]) for i in bans.find_all('span', class_= 'count')]

    ouput.reverse()

    return ouput



def loss_rate(iid):

    """calculate % missing for each rate

    return: dictionary with keys 1-10"""

    

    temp = record.loc[record.iid == eval(iid)].rate.value_counts().to_dict()

    temp_dict = {i+1:0 for i in range(10)}

    for key,value in temp.items():

        temp_dict[key]+=value

        

    for idx, val in enumerate(rating_status(iid)):

        temp_dict[idx+1] = 100 * (1 - temp_dict[idx+1]/val) if val != 0 else 0

    

    

    return temp_dict



def get_anime_list(pages):

    """get samples of anime subjects

    return: a list of string"""

    ouput = []

    for count in range(1,pages+1):

        temp = requests.get(mirror_bgm + f'anime/browser?sort=rank&page={count}')

        temp.encoding = "UTF-8"

        temp = temp.text

        bans = BeautifulSoup(temp, 'html.parser') 

        

        ouput += [i['href'][9:] for i in bans.find_all('a', class_ = "subjectCover cover ll")]

    return ouput
anime_list = get_anime_list(25)

data_dict = {i+1:[] for i in range(10)}

data_dict['iid'] = []

for iid in anime_list:

    temp = loss_rate(iid)

    for key,value in temp.items():

        data_dict[key].append(value)

    data_dict['iid'].append(iid)   

    

loss_df = pd.DataFrame(data_dict).set_index('iid')

loss_df.head(3)
loss_df.mean()
loss_df.std()
sns.set(rc={'figure.figsize':(10,6)})

loss_df.apply(np.mean, axis = 1).hist(bins = 25)#.mean()
print(f"平均丢失率为{round(loss_df.apply(np.mean, axis = 1).mean(),2)}%。")