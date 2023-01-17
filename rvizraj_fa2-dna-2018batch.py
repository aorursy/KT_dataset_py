# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

#SECTION 1

#Qn 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.plotly as py
import plotly
mu = 5
sigma = 2
s = np.random.normal(mu, sigma, 10000)
# Create the bins and histogram
count, bins, ignored = plt.hist(s, 20, normed=True)

# Plot the distribution curve
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    np.exp( - (bins - mu)**2 / (2 * sigma**2) ),       linewidth=3, color='y')
plt.figure(1)
plt.title('Question no 1')
plt.show()



#Qn 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
m=10
s=3
p=st.norm.ppf(1-0.67,scale=s,loc=m)
print("Question no 2")
print(p)


#Question No : 3
from scipy.stats import poisson
from scipy.stats import expon
Rate=3/2 #3 a
print("Question no 3a")
print(Rate)
p=expon.cdf(3,scale=1/Rate)
print("Question no 3b")
print(p)    #3 b
print("Question no 3c")
q=poisson.cdf(3,2*Rate) 
print(1-q)     #3c



# Question no : 4
from scipy.stats import f

dfn=4
dfd=5

a=f.ppf(.9, dfn, dfd)  #4 a
b=f.mean(dfn, dfd, loc=0, scale=1) # 4 b
print("Question no 4a")
print(a)
print("Question no 4b")
print(b)



# Qn 5
from scipy.stats import chi2
a=chi2.ppf(0.75, 24) #degree pf freedom=n-1
print("Question no 5")
print(a)



#Qn 6 # 
print("6a The disribution is bionomial distribution")
from scipy.stats import binom
print("Question no 6b")
binom.pmf(3, 7, 2/6)  #6 b




# Qn 7
# null hypothesis mu=32
from scipy.stats import t
mu=32 
Xbar=26
s= 6 
n=25
rootn=n**0.5
sigma=s/rootn
t_value=(Xbar-mu)/sigma
print("Question no 7")
print("Tvalue",t_value)
print("probability")
t.cdf(Xbar, n-1, loc=mu, scale=sigma)



#Question 8
unloc=pd.read_excel("../input/dataset/unloc.xlsx")
#list(unloc)
print("Question no 8 a")
df=unloc.groupby(['area_name'])['Low-income countries','Middle-income countries','High-income countries'].sum() 
print(df)
print("Question no 8 b")
unloc['Least or less developed countries']=unloc['Less developed countries']|unloc['Least developed countries'] #8 b
print(unloc['Least or less developed countries'])

unloc=unloc.dropna()
print("Question no 8 c")
unloc   #8 c

#Question no 8 d
plt.figure(2)
df.plot(kind='bar',title="Question no 8 d")



# Qn 8e

import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
unloc=pd.read_excel('../input/dataset/unloc.xlsx')
a1=unloc['Low-income countries'].sum()
a2=unloc['Middle-income countries'].sum()
a3=unloc['High-income countries'].sum()

# Data to plot
labels = 'Low-income countries', 'Middle-income countries', 'High-income countries'
sizes = [a1,a2,a3]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode 1st slice
 
# Plot
plt.figure(4)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Question no 8 e')
plt.show()













#Question 9
# This question will run only on Anaconda jupyter notebook, please import plotly and edit the details of username and API
#Part a
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly
import os
print(os.listdir("../input"))
df = pd.read_excel("../input/population1/population.xlsx")
df2 = pd.read_csv("../input/codes-country/2014_world_gdp_with_codes.csv")
df2=df2.rename(columns = {'COUNTRY':'name'})
df = pd.merge(df,df2[['name','CODE']],on='name', how='left')
df["pop_diff"] = np.sign(df['2015_male'] - df['2015_female'])

plotly.tools.set_credentials_file(username='kriswa', api_key='z0K0JEUQvSC6ExFVMmA5')
data = [ dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['pop_diff'],
        text = df['name'],
        colorscale = [[-1,"rgb(0, 0, 0)"],[0,"rgb(0, 0, 0)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,2555)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Poplulation Difference<br>Male - Female'),
      ) ]

layout = dict(
    title = 'Polpulation Map',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )

#Question 9 World Coud
#part b
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly
import os
print(os.listdir("../input"))

#import csv
population_df= pd.read_excel('../input/population1/population.xlsx', sheet_name='Sheet1')

#create new df with two columns
cloud_df=population_df[['name']].copy()
cloud_df['population percent']= population_df['2015_total']/population_df['2015_total'].sum()

#create dictionary
d = {}
for a, x in cloud_df.values:
    d[a] = x

#code for creating wordcloud
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#SECTION 2

#Qn 10 a data cleaning and analysis
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data_sch=pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")


#data cleaning 
data_sch.shape
data_sch[data_sch.iloc[:,:] == 'N/A'] = np.nan
data_s=data_sch.iloc[:,3:] #subsetting data to remove first 3 columns 
#data_sch[data_sch.isna()]

data_s=data_s.dropna()   # drop the columns with na values 
data_s=data_s.reset_index(drop=True) # reset the index 

#list(data_s)



# formatting data

a=list(data_s.columns[15:24])
for i in a:
    data_s[i] = data_s[i].str.rstrip('%').astype('float') / 100.0  # converting the percentage to float values 

data_s['School Income Estimate'] = data_s['School Income Estimate'].str.rstrip('$').replace(',','')


data_s['School Income Estimate'] = data_s['School Income Estimate'].str.replace('$','')
data_s['School Income Estimate'] = data_s['School Income Estimate'].str.replace(',','').astype('float')  #converting money table to float 


#data analysis 

data1=data_s.groupby(['Community School?'])['Percent ELL',
 'Percent Asian',
 'Percent Black',
 'Percent Hispanic',
 'Percent Black / Hispanic',
 'Percent White'].mean()
data2=data_s.groupby(['District'])['Economic Need Index'].mean()
data2=data2.to_frame()
data2.columns.values[0]='Economic Need Index'

data3=data_s.groupby(['District'])[
 'Percent Asian',
 'Percent Black',
 'Percent Hispanic',
 'Percent White'].mean()


data4=data_s.groupby(['District'])['School Income Estimate'].mean()
data4=data4.to_frame()
data4.columns.values[0]='School Income Estimate'



data5=data_s.groupby(['City'])['Percent Asian'].mean()
data5=data5.to_frame()
data5.columns.values[0]='Average Percentage of Asians'

data6=data_s.groupby(['Supportive Environment Rating'])['Percent of Students Chronically Absent'].mean()
data6=data6.to_frame()
data6.columns.values[0]='Percent of Students Chronically Absent'
 
print('yes')

# Question 10 visualisation and questions 

plt.figure(1)
#Question no 1 How does the distribution of students vary from community school to non community schools ?
data1.plot(kind='bar',figsize=(10,10),title='How does the distribution of students vary from community school to non community schools ?')

#Question no 2 How does  economic index  vary in the schools districtwise ?
plt.figure(2)

data2.plot(kind='bar',title='How does  economic index  vary in the schools districtwise ?')

plt.figure(3)
#Question no 3 How does the distribution of students vary districtwise?
data3.plot(kind='bar',figsize=(20,10),title='How does the distribution of students vary districtwise?')

plt.figure(5)
#Question no 4 How does  School Income Estimate  vary in the schools districtwise ?
data4.plot(kind='bar',figsize=(20,10),title='How does  School Income Estimate  vary in the schools districtwise ?')

plt.figure(7)
#Question no 5 How does  percentage of asians   vary in the schools citywise ?
data5.plot(kind='bar',figsize=(20,10),title='How does  percentage of asians   vary in the schools citywise ?')

plt.figure(9)
#Question no 6 How does  percentage of chronically absent students  vary with supportive environment ?
data6.plot(kind='bar',figsize=(20,10),title='How does  percentage of chronically absent students  vary with supportive environment ?')


#Question 11 data cleaning 

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import re

data_em=pd.read_csv("../input/montcoalert/911.csv")
#list(data_em)
data_em.shape


#data cleaning
#seperating types and subtypes in data set
data_em['title'],data_em['subtype']=data_em['title'].str.split(':',1).str                                   
data_em
#extracting date and time 

data_em['date'],data_em['time']=data_em['timeStamp'].str.split(':',1).str     
data_em['timeStamp'] = data_em['timeStamp'].astype("datetime64[ns]")
data_em['date'] = data_em['date'].astype("datetime64[ns]")

#extracting station number from the data and storing in datframe
df=pd.DataFrame()
regex_pat = re.compile(r'station', flags=re.IGNORECASE)
df['1'],df['2'],df['3'],df['4']=data_em['desc'].str.split(';',3).str
df=df[df['3'].str.contains(regex_pat)]
df['5'],df['3']=df['3'].str.split('Station',3).str
df['3']=df['3'].str.strip(":")
df['3']
data_em['station']=df['3'] #station numbers are saved for the data which are available 

data_em['lat']=data_em['lat'].astype('float')
list(data_em)

data7=data_em.groupby('title').size()

data8=data_em.groupby('station').size()

data=data_em.groupby(['lat','lng'])['lng'].count()
data=data.to_frame()
data.columns.values[0]='count'
data9=data[data['count']>=1000]

data10=data_em[data_em['title']=='EMS']

data11=data10.groupby(['twp'])['lng'].count()

data12=data_em.groupby([data_em['date'].dt.year, data_em['date'].dt.month])['lat'].count()
data12=data12.to_frame()
data12.columns.values[0]='Number of calls '
#Data visualisation and questions 

plt.figure(11)
#Question no 1 How does the distribution of calls vary typewise ?
data7.plot(kind='bar',figsize=(10,10),title='How does the distribution of calls vary typewise ?')

#Question no 2 How does  calls coming stationwise ?
plt.figure(12)

data8.plot(kind='bar',figsize=(30,30),title='How does  calls coming stationwise ?')

plt.figure(14)
#Question no 3 Which are location cordinates from which more than 1000 calls are coming?
data9.plot(kind='bar',figsize=(10,20),title='Which are location cordinates from which more than 1000 calls are coming?')

plt.figure(16)
#Question 4 Distribution of calls twp wise for all the calls came for EMS type
data11.plot(kind='bar',figsize=(30,30),title='Distribution of calls twp wise for all the calls came for EMS type')

plt.figure(18)
# Question 5 What is the distribution of calls monthwise
data12.plot(kind='bar',title='What is the distribution of calls monthwise')