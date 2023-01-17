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
import requests
from bs4 import BeautifulSoup
url = 'https://www.python.org/'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
elementos = soup.find_all('div', class_='small-widget')
for elemento in elementos:
    print(elemento.h2.text)
url = 'https://www.abimad.com.br/associados.php'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')
soup.find('figure')
empresa = soup.find('figure').h5.text
print(empresa)
telefone = soup.find('figure').find('i', class_='fa-phone').next_element.next_element.text
print(telefone)
link = soup.find('figure').find('i', class_='fa-link').next_element.next_element.text
print(link)
email = soup.find('figure').find('i', class_='fa-at').next_element.next_element.text
print(email)
todos_elementos = soup.find_all('figure')
empresa = []
telefone = []
site = []
email = []

for elemento in todos_elementos:
    empresa.append(elemento.h5.text)
    telefone.append(elemento.find('i', class_='fa-phone').next_element.next_element.text)
    site.append(elemento.find('i', class_='fa-link').next_element.next_element.text)
    email.append(elemento.find('i', class_='fa-at').next_element.next_element.text)  
df = pd.DataFrame({
                 'Empresa':empresa,
                 'Telefone':telefone,
                 'email':email,
                 'site':site
                    })
df.head()
#df.to_csv('lista.csv')