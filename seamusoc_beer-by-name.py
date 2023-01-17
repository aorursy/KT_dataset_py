""" loading libraries"""

from IPython.display import display, HTML



from datetime import datetime

from csv import DictReader, reader

import sys

import csv

from collections import defaultdict

%matplotlib inline

import csv

import pandas as pd    #pip install pandas

import matplotlib.pyplot as plt

import numpy as np

import re



pd_beer= pd.read_csv('../input/beers.csv')

pd_brew = pd.read_csv('../input/breweries.csv')

pd_beer.drop('Unnamed: 0',axis=1,inplace =True)

 
pd_beer['words']=pd_beer['name'].apply(lambda x :len(str.split(x,' ')))

pd_beer['words'].hist(bins=9)
pd_long = pd_beer[pd_beer.words >7] 

display(HTML(pd_long[['name','style']].to_html(index=False)))
pd_beer['letter_count']=pd_beer['name'].apply(lambda x :len(x))

pd_beer['letter_count'].hist(bins=50,histtype ='stepfilled',color='green',label='Histogram of characters in beer name')
pd_short_beer=pd_beer[pd_beer['letter_count']<=3]

pd_short_beer[ [ 'name','style'] ]





display(HTML(pd_short_beer[['name','style']].to_html(index=False)))
all_words = []

for a,b in pd_beer.name.iteritems():

    x = b.split(' ')

    all_words.extend(x) 

         



# non ascii characters 

print ("Non ascii characters in the names:")

unusual_chars=set()

for x in all_words:

    res  =re.sub('[0-9a-zA-Z()]+', '', x)

    if len(res)>0:

        

        

       unusual_chars.add(res) 

print  (unusual_chars )  

print ("\n" )       



res = pd.DataFrame()

for x in unusual_chars:

    if len(x)>0:    

      res =  pd_beer[pd_beer.name.str.contains(re.escape(str(x)))]

      print (x,' found ',len(res),' times')

      #print (res[['name']])

  

   

        

        

       
unlauts_and_such ={ 'è', 'ä', 'ö', 'é', 'í', '™', 'ü', '°'}



res = pd.DataFrame()

for x in unlauts_and_such:

    if len(x)>0:    

      res =  res.append (pd_beer[pd_beer.name.str.contains(re.escape(str(x)))])

       

display(HTML(res[['name','style']].to_html(index=False)))
 

pd_brew['words']=pd_brew['name'].apply(lambda x :len(str.split(x,' ')))

short_brew = pd_brew [pd_brew['words']==1]





pd_brew['words'].hist(bins=6)

display(HTML(short_brew[['name','city']].to_html(index=False)))





pd_beer.corr()