# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import requests 
from bs4 import BeautifulSoup
url='https://www.imobiliare.ro/inchirieri-apartamente/targu-mures'
response=requests.get(url)
soup=BeautifulSoup(response.text, 'html.parser')
#ár
results1=soup.find_all('div', attrs={'class':'pret'})
#cím
results2=soup.find_all('div', attrs={'class':'localizare'})
#tulajdonságok
results3=soup.find_all('div', attrs={'class':'col-lg-12 col-md-12 col-sm-12 col-xs-12 no-padding-left text-center'})
results1[0]
results1[0].get_text().strip()
c1=[]

for i in range(len(results1)):
        text=results1[i].get_text().strip()
        c1.append(text)
c2=[]

for i in range(len(results2)):
    text=results1[i].get_text().strip()
    c2.append(text)
        
c3=[]

for i in range(len(results3)):
    text=results3[i].get_text().strip()
    c3.append(text)
oldal_linkek=[]
oldal_linkek.append('https://www.imobiliare.ro/inchirieri-apartamente/targu-mures')
for x in range(2,10):
    url='https://www.imobiliare.ro/inchirieri-apartamente/targu-mures?pagina='+str(x)+''
    oldal_linkek.append(url)
c1=[]
c2=[]
c3=[]

for n in range(len(oldal_linkek)):
    url=oldal_linkek[n]
    response=requests.get(url)
    soup=BeautifulSoup(response.text, 'html.parser')
    
    #ár
    results1=soup.find_all('div', attrs={'class':'pret'})
    #cím
    results2=soup.find_all('div', attrs={'class':'localizare'})
    #tulajdonságok
    results3=soup.find_all('div', attrs={'class':'col-lg-12 col-md-12 col-sm-12 col-xs-12 no-padding-left text-center'})
    
    for i in range(len(results1)):
        text=results1[i].get_text().strip()
        c1.append(text)
    for i in range(len(results2)):
        text=results2[i].get_text().strip()
        c2.append(text)
    for i in range(len(results3)):
        text=results3[i].get_text().strip()
        c3.append(text)
        
        
tabla_alberlet={"Ár": c1,
                "Cím": c2,
                "Tulajdonságok": c3
               }
df_alberlet=pd.DataFrame.from_dict(tabla_alberlet)

writer=pd.ExcelWriter('alberletek.xlsx', engine='xlsxwriter')
df_alberlet.to_excel(writer, sheet_name='Albérletek Marosvásárhelyen', index=False)
writer.save()





