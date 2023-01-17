# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_excel("/kaggle/input/covid19-hitting-tunisia/tounes.xlsx")

data.head()
data.isna().sum() 
import plotly.express as px

date=data['Gouvernorat'].value_counts().reset_index()

px.bar_polar(date, theta="index", r="Gouvernorat", template="plotly_dark",

            color_discrete_sequence= px.colors.sequential.Plasma_r)

dates=data['Date de confirmation'].value_counts().reset_index().rename(columns={'index': 'Date de confirmation' ,'Date de confirmation':' cas'}).sort_values(by = 'Date de confirmation')

px.line(dates, x='Date de confirmation', y=' cas')

age=data.Age.value_counts().reset_index().rename(columns={'index': 'Age' ,'Age':' cas'})

import plotly.express as px

px.bar(age , y=' cas' , x='Age')
#data['Gouvernorat']=data['Gouvernorat'].replace({'Ariana':"TN-12",'Tunis':"TN-11", "Ben Arous": "TN-13","Sousse": "TN-51", "Tataouine":"TN-83","Mednine": "TN-82" , "Manouba": "TN-14","Kebili": "TN-73","Monastir":"TN-52", "Sfax":"TN-61", "Mahdia":"TN-53","Kairouan":"TN-41","Bizerte":"TN-23","Gabes": "TN-81", "Gafsa":"TN-71","Nabeul":"TN-21" })

sexe=data.Genre.value_counts().reset_index().rename(columns={'index': 'Sexe' ,'Genre':' cas'})

px.pie(sexe, names='Sexe', values=' cas')