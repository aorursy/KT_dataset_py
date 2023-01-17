from collections import Counter

from datetime import timedelta, datetime

import glob

from itertools import chain

import json

import os

import re

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas.plotting import register_matplotlib_converters

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

from tqdm import tqdm_notebook

# font_path = '/usr/share/fonts/NanumGothic.ttf'

# font_name = fm.FontProperties(fname=font_path, size=10).get_name()

# plt.rc('font', family=font_name, size=12)

plt.rcParams["figure.figsize"] = (20, 10)

register_matplotlib_converters()
path1 = '../input/kakao-arena-2nd-competition/'

path_meta = '../input/kakao-arena-eda-metadata/'



# magazine = pd.read_json(path1 + 'magazine.json', lines=True)

# metadata = pd.read_json(path1 + 'metadata.json', lines=True)

# metadata2 = pd.read_csv(path_meta + 'metadata.csv')



users = pd.read_json(path1 + 'users.json', lines=True)



dev = pd.read_csv(path1+'predict/predict/dev.users', names=['id'])

test = pd.read_csv(path1+'predict/predict/test.users', names=['id'])
path2 = '../input/kakao-arena-2nd-competition/read/read/'

read_file_lst = os.listdir(path2)

exclude_file_lst = ['read.tar', '.2019010120_2019010121.un~']



read_df_lst = []

for f in read_file_lst:

    file_name = os.path.basename(f)

    if file_name in exclude_file_lst:

        print(file_name)

    else:

        df_temp = pd.read_csv(path2+f, header=None, names=['raw'])

        df_temp['dt'] = file_name[:8]

        df_temp['hr'] = file_name[8:10]

        df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]

        df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()

        read_df_lst.append(df_temp)

read = pd.concat(read_df_lst)

read = read[read['article_id']!='']
users.to_csv('users.csv', index=False)

dev.to_csv('dev.csv', index=False)

test.to_csv('test.csv', index=False)

read.to_csv('read.csv', index=False)