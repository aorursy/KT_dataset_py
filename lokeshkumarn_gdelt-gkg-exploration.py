import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math

import gdelt as gd



import matplotlib.pyplot as plt
gs = gd.gdelt(version=2)
df=gs.Search(date=['2019 04 01','2019 04 03'],table='gkg',output='df',normcols=True)
def checkFloatNan(x):   

    if type(x).__name__ == 'float':

        return math.isnan(x)

    else:

        return False        

    
df['IsThemeNaN'] = df['themes'].apply(checkFloatNan)



df = df[df['IsThemeNaN']==False][['gkgrecordid', 'date', 'sourcecollectionidentifier', 'sourcecommonname',

       'documentidentifier', 'counts', 'v2counts', 'themes', 'v2themes',

       'locations', 'v2locations', 'persons', 'v2persons', 'organizations',

       'v2organizations', 'v2tone', 'dates', 'gcam', 'sharingimage',

       'relatedimages', 'socialimageembeds', 'socialvideoembeds', 'quotations',

       'allnames', 'amounts', 'translationinfo', 'extras']]
df.head()
df.shape
print(df['sourcecollectionidentifier'].value_counts())



print('1 = WEB')
df['sourcecommonname'].value_counts()[:5]#Web Domain names
df['documentidentifier'][:5]#Complete Urls
df['documentidentifier'][:1][0]
print(df[~df['counts'].isnull()]['counts'][2:5].reset_index(drop=True)[0])

print('-------------')

print(df[~df['v2counts'].isnull()]['v2counts'][2:5].reset_index(drop=True)[0])
df['themes'][0].split(';')
df['v2themes'][0].split(';')#CharOffset
print(df[~df['locations'].isnull()]['locations'].reset_index(drop=True)[0])

print('-----------------')

print(df[~df['v2locations'].isnull()]['v2locations'].reset_index(drop=True)[0])
print(df[~df['persons'].isnull()]['persons'][:5].reset_index(drop=True)[0])

print('--------------')

print(df[~df['v2persons'].isnull()]['v2persons'][:5].reset_index(drop=True)[0])#Char Offset
print(df[~df['organizations'].isnull()]['organizations'].reset_index(drop=True)[0])

print('----------------')

print(df[~df['v2organizations'].isnull()]['v2organizations'].reset_index(drop=True)[0])#Char Offset
df['v2tone'].reset_index(drop=True)[0]#between -10 and +10,Positive Score 0-100,Negative Score 0- 100,Polarity score,Ref Den,Ref Den,Word Count.
df['sharingimage'].reset_index(drop=True)[1]#image
df[~df['relatedimages'].isnull()]['relatedimages'].reset_index(drop=True)[0]
df[~df['socialimageembeds'].isnull()]['socialimageembeds'].reset_index(drop=True)[0]
df[~df['socialvideoembeds'].isnull()]['socialvideoembeds'].reset_index(drop=True)[0]
df[~df['quotations'].isnull()]['quotations'].reset_index(drop=True)[0]#OFfset|length|verb|Actual Quote
df[~df['allnames'].isnull()]['allnames'].reset_index(drop=True)[0]#names#offset
df[~df['amounts'].isnull()]['amounts'].reset_index(drop=True)[0]#Amount,object,offset
df[~df['translationinfo'].isnull()]['translationinfo']#blank for documents originally in English

#SRCLC. This is the Source Language Code,ENG. This is a textual citation string that indicates the engine(
df['extras'].reset_index(drop=True)[1]#XML