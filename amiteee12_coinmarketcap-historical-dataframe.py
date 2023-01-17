import numpy as np

import pandas as pd

from pandas.io.json import json_normalize



from requests import Request, Session

from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

import json



from bs4 import BeautifulSoup

import requests



from tqdm import tqdm_notebook as tqdm

import time
def get_data(c,start='20130428',end='20190528'):

    url = 'https://coinmarketcap.com/currencies/'+c+'/historical-data/?start='+start+'&end='+end

    url=url.replace(' ','-')

    content = requests.get(url).content

    soup = BeautifulSoup(content,'html.parser')



    #         time.sleep(1)



    table = soup.find('table', {'class': 'table'})

    data = [[td.text.strip() for td in tr.findChildren('td')] 

            for tr in table.findChildren('tr')]

    df = pd.DataFrame(data)

    df.drop(df.index[0], inplace=True) # first row is empty

    df[0] =  pd.to_datetime(df[0]) # date

    for i in range(1,7):

        df[i] = pd.to_numeric(df[i].str.replace(",","").str.replace("-","")) # some vol is missing and has -

    df.columns = ['Date','Open','High','Low','Close','Volume','Market Cap']

    df['Name']=c

    return df

df_total=pd.DataFrame() 

df_list=[]

for c in tqdm(['bitcoin-cash','bitcoin','dash','dogecoin','ethereum','iota','litecoin','nem','neo']):

    print(c)

    try:

        df_tmp=get_data(c)

        df_list.append(df_tmp)

    except:

        print('failed to parse for :%s'%(c))

        

# df_total=pd.concat(df_list)

# df_total=df_total.sort_values(by=['Name','Date']).reset_index()

# df_total

print(len(df_list))
df_total=pd.concat(df_list)

df_total=df_total.sort_values(by=['Name','Date']).reset_index(drop=True)

df_total.to_csv('crypto_amit_may28.csv', index=False)
df_total[df_total.Name=='bitcoin'].reset_index()['Market Cap'].plot()

df_total[df_total.Name=='ethereum'].reset_index()['Market Cap'].plot()
print(df_total[df_total.Name=='bitcoin'].shape)

print(df_total[df_total.Name=='ethereum'].shape)
!ls 
df_total