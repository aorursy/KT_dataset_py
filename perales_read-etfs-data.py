import pandas as pd

import zipfile

import re

import glob
path = '../input/Data.zip' #Data.zip folder

zf = zipfile.ZipFile(path) 

folder = 'ETFs' 


filelist = []

for i in zf.namelist():

    '''

    Take only the files which belong to the folder previously stored.

    '''

    if re.match(folder+'*', i):

        filelist.append(i)

#Take the name of the first ticker

filelist[0].split('/')[1].split('.')[0]
#Loop which list the tickers' name from the filelist

tickers = []



for i in filelist:

    tickers.append(i.split('/')[1].split('.')[0]) #We want the tickers name which is after the / symbol.

    
list_ = []



df = pd.DataFrame()



for i in filelist:

    

    df_a = pd.read_csv(zf.open(i),index_col=0)

    list_.append(df_a)

df = pd.concat(list_, axis=1, keys= tickers)

df.tail()