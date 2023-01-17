import os

os.listdir('../input/')
# importuri



# data download

import urllib.request, json 



# procesare

import pandas as pd

import numpy as np

import sklearn

import scipy



# vizualizare

import matplotlib.pyplot as plt

import seaborn as sns

sns.set() # aia e, imi place mai mult
# In primul rand downloadez datele din linkul "https://covid19.geo-spatial.org/api/dashboard/getDailyCaseReport"



with urllib.request.urlopen("https://covid19.geo-spatial.org/api/dashboard/getDailyCaseReport") as url:

    data = json.loads(url.read().decode())

    

# inghesui totul intr-un dataframe citibil



cz = pd.DataFrame(data['data'].values()).T[0].apply(pd.Series)

cz.day_case = pd.to_datetime(cz.day_case)



# si vad ce-a iesit



cz.head()
# sa vedem daca avem valori lipsa



cz.isnull().sum()
cz.cov()
cz.corr()
corr_mat = cz.corr()

f, ax = plt.subplots()

sns.heatmap(corr_mat, ax = ax, linewidths = 0.1)
sns.jointplot(x = 'new_case_no', y = 'new_healed_no', data = cz, color = 'green', kind='reg') # aka regresie
sns.jointplot(x = 'total_case', y = 'total_dead', data = cz, color = 'green', kind='reg')
sns.jointplot(x = 'new_case_no', y = 'new_dead_no', data = cz, color = 'green', kind='reg')
# de aici: https://docs.google.com/spreadsheets/d/1tLDr4oFSFl8Gjjk8rtKJBCy2oimF5zC_eAJZMxBXLK8/edit#gid=465341240

suceava = pd.read_excel('../input/jurnal_Suceava.xlsx') 

suceava.head(2)
suceava_nou = suceava.loc[suceava['Dată deces'].notnull()]

len(suceava_nou), suceava_nou['Afectiuni preexistente'].isnull().sum(), 83-34
print(suceava_nou['Afectiuni preexistente'].value_counts()[:5])
# de aici: https://docs.google.com/spreadsheets/d/12iMFwXZpKjeLEz_YqS7gRbLytAzNi45cAC2CD741Mys/edit#gid=465341240



bucuresti = pd.read_excel('../input/jurnal_Bucuresti.xlsx')

bucuresti_nou = bucuresti.loc[bucuresti['Dată deces'].notnull()]

len(bucuresti_nou), bucuresti_nou['Afectiuni preexistente'].isnull().sum(), 32-4
bucuresti_nou['Afectiuni preexistente'].value_counts()[:5]
text_bucuresti = str(bucuresti_nou['Afectiuni preexistente'].values)

text_suceava = str(suceava_nou['Afectiuni preexistente'].values)

text = text_bucuresti + text_suceava

text = text.lower()

text = text.replace('nan','')

text = text.replace('ă','a')
from wordcloud import WordCloud

wordcloud = WordCloud().generate(text)



fig = plt.figure(figsize = (10.5, 7.2))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#wordcloud.to_file('covid.png')
#  https://covid19.geo-spatial.org/api/dashboard/getDailyCases



with urllib.request.urlopen("https://covid19.geo-spatial.org/api/dashboard/getDailyCases") as url:

    data = json.loads(url.read().decode())

    

# inghesui totul intr-un dataframe citibil



dc = pd.DataFrame(data['data'].values()).T[0].apply(pd.Series)

dc.Data = pd.to_datetime(dc.Data)



# si vad ce-a iesit



dc.columns
dc1 = dc[dc['Nr de teste pe zi'] > 0]



plt.figure(figsize = (10.5,7.2))

plt.plot(dc1.Data, dc1['Nr de teste pe zi'], label='Numar de teste')

plt.plot(dc1.Data, dc1['Cazuri'], label='Numar de cazuri')

plt.plot(dc1.Data, dc1['Terapie intensiva'], label='Terapie intensiva')

plt.legend()

plt.show()
dc1.corr()
dc['Nr de teste pe zi'].isnull().sum()
fig = plt.figure(figsize = (10.5, 7.2))

plt.bar('day_case', 'new_dead_no', data = cz, 

        color = 'blue', edgecolor = 'white', width = 1)

plt.bar('day_case', 'total_dead', data = cz, 

        bottom = 'new_dead_no', color = 'darkorange', edgecolor = 'white', width = 1)

plt.title('Decese')

plt.legend(['new_dead_no', 'total_dead'])

plt.xticks(rotation=60)

cz.new_dead_no.hist(bins = 20)
cz1 = cz[cz.total_dead > 0]

cz1.new_dead_no.hist(bins = 10)
fig = plt.figure(figsize = (10.5, 7.2))

plt.bar('day_case', 'new_dead_no', data = cz1, 

        color = 'blue', edgecolor = 'white', width = 1)

plt.bar('day_case', 'total_dead', data = cz1, 

        bottom = 'new_dead_no', color = 'darkorange', edgecolor = 'white', width = 1)

plt.title('Decese')

plt.legend(['new_dead_no', 'total_dead'])

plt.xticks(rotation=60)
fig = plt.figure(figsize = (10.5, 7.2))

plt.bar('day_case', 'new_dead_no', data = cz, 

        color = 'blue', edgecolor = 'white', width = 1)

plt.bar('day_case', 'new_case_no', data = cz, 

        bottom = 'new_dead_no', color = 'darkorange', edgecolor = 'white', width = 1)

plt.title('Decese / Cazuri')

plt.legend(['new_dead_no', 'new_case_no'])

plt.xticks(rotation=60)
fig = plt.figure(figsize = (10.5, 7.2))

sns.heatmap(cz1.drop(columns=['day_case']))