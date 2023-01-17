# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path2 = '../input/read/read/'

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



# read data cleaning



from itertools import chain

def chainer(s):

    return list(chain.from_iterable(s.str.split(' ')))

read_cnt_by_user = read['article_id'].str.split(' ').map(len)

read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),

                         'hr': np.repeat(read['hr'], read_cnt_by_user),

                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),

                         'article_id': chainer(read['article_id'])})

read_raw = read_raw.reset_index(drop=True)

read_raw['article'] = read_raw['article_id'].apply(lambda x: str(x).split('_')[0])

del read



read_raw2 = read_raw[read_raw['dt']>='20190222'].reset_index(drop=True)

read_raw2.drop(columns=['dt', 'hr'], inplace=True)

read_raw2['article'] = read_raw2['article_id'].apply(lambda x: str(x).split('_')[0])
path1='../input/'
metadata = pd.read_json(path1 + 'metadata.json', lines=True)



# metadata preprocess

metadata = metadata[['id', 'user_id', 'keyword_list']]

metadata = metadata[metadata['id'].isin(read_raw2['article_id'])]

metadata = metadata.loc[metadata['keyword_list'].apply(lambda x: x if x!=[] else np.nan).dropna().index]
metadata2 = pd.DataFrame()

for idx in metadata.index:

    temp = metadata.loc[idx]

    keyword_list = temp['keyword_list']

    temp = pd.DataFrame([temp for i in range(len(keyword_list))])

    temp['keyword_list'] = keyword_list

    metadata2 = pd.concat([metadata2, temp])
metadata2
metadata2.reset_index(drop=True).to_csv('metadata.csv', index=False)