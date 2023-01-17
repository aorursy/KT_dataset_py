import numpy as np 

import pandas as pd 

import os

import tqdm
os.listdir('../input/')
path = '../input/coronavirus-covid19-tweets/'

path1 = '../input/coronavirus-covid19-tweets-early-april/'

path2 = '../input/coronavirus-covid19-tweets-late-april/'
fname = os.listdir(path)

fname1 = os.listdir(path1)

fname2 = os.listdir(path2)
def get_data(df, output):

    for i in tqdm.tqdm(range(len(df))):

        try:

            if(df.iloc[i]['country_code'] == 'MY'):

                line = [df.iloc[i]['text'], df.iloc[i]['retweet_count'], df.iloc[i]['favourites_count']

                        , df.iloc[i]['created_at'], df.iloc[i]['lang']]

                output.append(line)

        except Exception as e:

            pass

    return output
df_MY = []
for n in fname:

    print("Processing: {}".format(n))

    df = pd.read_csv(os.path.join(path, n))

    df_MY = get_data(df, df_MY)
for n in fname1:

    print("Processing: {}".format(n))

    df = pd.read_csv(os.path.join(path1, n))

    df_MY = get_data(df, df_MY)
# Get data

for n in fname2:

    print("Processing: {}".format(n))

    df = pd.read_csv(os.path.join(path2, n))

    df_MY = get_data(df, df_MY)
len(df_MY)
df_MY_final = pd.DataFrame(df_MY, columns=['text', 'retweet_count', 'favourites_count', 'created_at', 'lang'])
df_MY_final.iloc[0:50]
df_MY_final.to_csv('COVID_tweet_MY.csv', index=False)