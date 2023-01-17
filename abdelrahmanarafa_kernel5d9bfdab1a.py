import requests

import lxml.html as lh

import pandas as pd

import numpy as np
url='https://www.espncricinfo.com/ci/content/player/25056.html'

#Create a handle, page, to handle the contents of the website

page = requests.get(url)

#Store the contents of the website under doc

doc = lh.fromstring(page.content)

#Parse data that are stored between <tr>..</tr> of HTML

tr_elements = doc.xpath('//tr')
#Check the length of the first 12 rows

[len(T) for T in tr_elements[:12]]
tr_elements = doc.xpath('//tr')

#Create empty list

col=[]



#For each row, store each first element (header) and an empty list

for t in tr_elements[0]:

    

    name=t.text_content()

    print(name)

    col.append(name)
raw=[]

for j in range(1,len(tr_elements)):

    T=tr_elements[j]

    print(len(T))    

    if (len(T)==14)  :

        break  

    elif (len(T)==15)  :

        for t in T.iterchildren():

            data=str(t.text_content())

            raw.append(data)
arr = np.array(raw)

n_raw=int(arr.shape[0]/15)

arr=arr.reshape(n_raw,15)

Bat=pd.DataFrame(data=arr,columns=col)

                    
Bat
tr_elements = doc.xpath('//tr')

#Create empty list

col=[]



#For each row, store each first element (header) and an empty list

for t in tr_elements[len(Bat.index)+1]:

    

    name=t.text_content()

    print(name)

    col.append(name)
raw1=[]

for j in range(1,len(tr_elements)):

    T=tr_elements[j]

    print(len(T))    

    if (len(T)==2)  :

        break  

    elif (len(T)==14)  :

        for t in T.iterchildren():

            data=str(t.text_content())

            raw1.append(data)
arr1 = np.array(raw1)

n_raw1=int(arr1.shape[0]/14)

arr1=arr1.reshape(n_raw1,14)
table1=pd.DataFrame(data=arr1,columns=col)

table1.shape
col[0]='sort'

df_Batting=pd.DataFrame(columns=col)
df_Batting
def get_data( url ):

    url=url

    page = requests.get(url)

    doc = lh.fromstring(page.content)

    tr_elements = doc.xpath('//tr')

    raw1=[]

    for j in range(1,len(tr_elements)):

        T=tr_elements[j]    

        if (len(T)==2)  :

            break  

        elif (len(T)==14)  :

            for t in T.iterchildren():

                data=str(t.text_content())

                raw1.append(data)

    arr1 = np.array(raw1)

    n_raw1=int(arr1.shape[0]/14)

    arr1=arr1.reshape(n_raw1,14)

    

    

    Bat=pd.DataFrame(data=arr1,columns=col)

    for i in Bat['sort'].values:

        if i =='ODIs':

             Bat= Bat.set_index([Bat.columns[0]])

             global df_Batting

             df_Batting=df_Batting.append(Bat.loc['ODIs'])

        else:

            continue
# To run this, download the BeautifulSoup zip file

# http://www.py4e.com/code3/bs4.zip

# and unzip it in the same directory as this file



#these are list of countries who has players of ODIs

countries={'1':'england','2':'australia','3':'southafrica','4':'westindies','5':'newzealand','6':'india','7':'pakistan','8':'srilanka','9':'zimbabwe','11':'usa','12':'Bermuda','14':'East and Central Africa','15':'Netherlands','17':'Canada','19':'Hong Kong','20':' Papua New Guinea','25':'Bangladesh','26':' Kenya','27':'United Arab Emirates','28':'Namibia','29':'Ireland','30':'Scotland','32':'Nepal','37':'Oman','40':'Afghanistan'}



from urllib.request import urlopen

from bs4 import BeautifulSoup

import ssl



# Ignore SSL certificate errors

ctx = ssl.create_default_context()

ctx.check_hostname = False

ctx.verify_mode = ssl.CERT_NONE

li=[]

for key,value in countries.items():

 url = 'https://www.espncricinfo.com/ci/content/player/index.html?country='+key

 html = urlopen(url, context=ctx).read()

 soup = BeautifulSoup(html, "html.parser")



# Retrieve all of the anchor tags

 tags = soup('a')

 count=0



 for tag in tags:

     tag1 =tag.get('href', None).split('.')

     tag2=tag1[0].split('/')

     for i in tag2:

            if i.isdigit()==True:

                li.append('https://www.espncricinfo.com/'+value+'/content/player/'+i+'.html') 







li = list(set(li)) 

print(len(li))



for i in li:

      get_data(i)

df_Batting.drop(df_Batting.columns[0],axis=1)



df_Batting


