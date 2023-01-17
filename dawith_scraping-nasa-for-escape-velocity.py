%matplotlib inline

import numpy as np 

import matplotlib.pyplot as plt 

from math import *

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
G = 6.674 *10**(-11)

M_e = 5.97*10**(24)

R_e = 6.37*10**6



def escape_velocity():

    print (sqrt(2*G*M_e/R_e))
escape_velocity()
import requests #library used to download web pages.

#specify the url

URL = "https://nssdc.gsfc.nasa.gov/planetary/factsheet/planet_table_ratio.html"
# The GET method indicates that you’re trying to get or retrieve data from a specified resource. 

# Connect to the website using the variable 'page'

# To make a GET request, invoke requests.get().

page = requests.get(URL)
# A Response is a powerful object for inspecting the results of the request.

type(page)
# verify successful connection to website



# To know about the all codes 

# https://www.restapitutorial.com/httpstatuscodes.html

  

#  a 200 OK status means that your request was successful,and the server responded with the data you were requesting,

# whereas a 404 NOT FOUND status means that the resource you were looking for was not found.     

page.status_code
#save string format of website HTML into a variable

HTMLstr = page.text

print(HTMLstr[:1000])
#import the Beautiful soup functions to parse the data returned from the website



# Beautiful Soup is a library that makes it easy to scrape information from web pages. It sits atop an HTML

# or XML parser, providing Pythonic idioms for iterating, searching, and modifying the parse tree.

from bs4 import BeautifulSoup



# parse the html using beautiful soup and store in variable `soup`

# First argument: It is the raw HTML content.

# Second Argument:  Specifying the HTML parser we want to use.

soup = BeautifulSoup(HTMLstr, "html.parser")
# soup.<tag>: Return content between opening and closing tag including tag.

soup.title
#shows the first <a> tag on the page

soup.a
#show hyperlink reference for all <a> tags

all_links=soup.find_all("a")



#The following selects the first 20 rows

# If you want to see all, remove [0:20]

for link in all_links[0:20]:

    print (link.get("href"))
all_tables=soup.find_all('table')

planet_table=soup.find('table')
#set empty lists to hold data of each column

A=[]#Physical Quantities

B=[]#Mercury

C=[]#Venus

D=[]#Earth

E=[]#Moon

F=[]#Mars

G=[]#Jupiter

H=[]#Saturn

I=[]#Uranus

J=[]#Neptune

K=[]#Pluto



#find all <tr> tags in the table and go through each one (row)

# tr table row tag

for row in planet_table.findAll("tr"):

    body=row.findAll('td') #To store second column data

    #get all the <td> tags for each <tr> tag

    cells = row.findAll('td')

    

    #if there are 11 <td> tags, 11 cells in a row

    if len(cells)==11: 

        

        A.append(cells[0].find(text=True)) #gets info in first column and adds it to list A

        B.append(cells[1].find(text=True)) # gets info of Mercury column and adds it to list B

        C.append(cells[2].find(text=True)) # gets info of Venus column and add it to list C

        D.append(cells[3].find(text=True)) # gets info of Earth and adds it to list D

        E.append(cells[4].find(text=True)) # gets info of Moon column and adds it to list E

        F.append(cells[5].find(text=True)) # gets info of Mars column and adds it to list F

        G.append(cells[6].find(text=True)) # gets info of Jupiter column tand adds it to list G

        H.append(cells[7].find(text=True)) # gets info of Saturn column tand adds to list H

        I.append(cells[8].find(text=True)) # gets info of Uranus column and adds it to list I

        J.append(cells[9].find(text=True)) # gets info of Neptune column and adds it to list J

        K.append(cells[10].find(text=True)) # gets info of NePluto column and adds to list K
#verify data in list A

A
#import pandas to convert list to data frame

import pandas as pd



df=pd.DataFrame(A, columns=['Physical_Measurement']) #turn list A into dataframe first



#add other lists as new columns in my new dataframe

df['Mercury'] = B

df['Venus'] = C

df['Earth'] = D

df['Moon'] = E

df['Mars'] = F

df['Jupiter'] = G

df['Saturn'] = H

df['Uranus'] = I

df['Neptune'] = J

df['Pluto'] = K

df=df.fillna(0)

df=df.replace(to_replace = 'Unknown*', value =0) 

#Planetary Fact Sheet - Ratio to Earth Values



#show first 5 rows of created dataframe

df

df=df.drop(df.index[0])

df=df.drop(df.index[-1])

df
df.dtypes
#getting rid of *

df= df.applymap(lambda x: x.strip('*') if isinstance(x, str) else x)

df

#display the table if everything is OK


df = df.replace(to_replace=['No', 'Yes','Unknown'], value=[0, 1, 2])

df
#change index

df = df.set_index('Physical_Measurement')

df
df.dtypes
df = df.apply(pd.to_numeric, errors='coerce')

df.dtypes
df
#Transpose index and columns.

#Reflect the DataFrame over its main diagonal by 

#writing rows as columns and vice-versa. 

#The property T is an accessor to the method transpose().

dfT = df.T

#check if columns are changed

dfT.columns

#Extract relevant columns 

dfT = dfT[['Mass', 'Diameter', 'Density', 'Gravity', 'Escape Velocity',

       'Rotation Period', 'Length of Day', 'Distance from Sun', 'Perihelion',

       'Aphelion', 'Orbital Period', 'Orbital Velocity']]

dfT
ax=dfT['Mass'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='lightgreen')

ax.set_xlabel("Planets")

ax.set_ylabel("Mass");
ax=dfT['Mass'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='lightgreen')

ax.set_xlabel("Planets")

#ax1.set_xlim([begin, end])

ax.set_ylim([0, 2])

ax.set_ylabel("Mass");
ax=dfT['Escape Velocity'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='g')

ax.set_xlabel("Planets")

ax.set_ylabel("Escape Velocity");
ax=dfT['Length of Day'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='m')

ax.set_xlabel("Planets")

ax.set_ylabel("Length of Day");
ax=dfT['Gravity'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='r')

ax.set_xlabel("Planets")

ax.set_ylabel("Gravity");
ax=dfT['Density'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='darkblue')

ax.set_xlabel("Planets")

ax.set_ylabel("Density");
ax=dfT['Distance from Sun'].plot(kind='bar',x=dfT.index, figsize=(10,7),color='b')

ax.set_xlabel("Planets")

ax.set_ylabel("Distance from Sun ");
dfT1 = pd.DataFrame(dfT, columns=['Gravity', 'Density', 'Escape Velocity'])

colors=['red','blue','lightgreen']

ax=dfT1.plot.bar( figsize=(12,8), color=colors)

ax.set_xlabel("Planets")

ax.set_ylabel("Gravity, Density OR Escape Velocity");
dfT2 = pd.DataFrame(dfT, columns=['Distance from Sun', 'Length of Day'])

ax=dfT2.plot.bar(figsize=(10,7))

ax.set_xlabel("Planets")

ax.set_ylabel("Distance from Sun OR Length of Day");
dfT3 = pd.DataFrame(dfT, columns=['Gravity', 'Density', 'Escape Velocity','Distance from Sun', 'Length of Day','Mass'])

ax=dfT3.plot.bar(stacked=True, figsize=(10,7),colormap='hsv',title='Defferent Physical Measurements of each Planet')

ax.set_xlabel("Planets")

ax.set_ylabel("Physical Parameters");


scaler = MinMaxScaler()

dfT4 = pd.DataFrame(scaler.fit_transform(dfT3), columns=dfT3.columns)
ax=dfT4.plot.bar(stacked=True, figsize=(10,7),colormap='rainbow',title='Defferent Physical Measurements of each Planet')

ax.set_xlabel("Planets")

ax.set_ylabel("Physical Parameters");