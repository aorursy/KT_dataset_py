# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, sep="\t", encoding = "ISO-8859-1")

print(df.dtypes)

df.head()
df = df.rename(columns = {"Deprem Kodu": "Earthquake Code", "Olus tarihi":"Date of occurrence", "Olus zamani" : "Time of occurrence", "Enlem":"Latitude", "Boylam":"Longitude", "Der(km)":"Depth(km)","Tip":"Type", "Yer":"Location"})
df.head(5)
df['Date of occurrence'] = pd.to_datetime(df['Date of occurrence'], format = '%Y.%m.%d') 
df['Date of occurrence'] = df['Date of occurrence'].dt.strftime("%Y-%m-%d")
df['datetimestr'] = df['Date of occurrence'] + ' ' + df['Time of occurrence'].str.slice(0,5) 

# I ignore the second part
df['datetime'] = pd.to_datetime(df['datetimestr'])
df['Time of occurrence'].str.slice(6,8).unique()
df.head()
df['Type'].value_counts()
df.plot(x = 'datetime', y = 'Depth(km)')
df.plot(x = 'datetime', y = 'xM')
df.plot(x = 'datetime', y = 'MD')

plt.title("Duration")
df.plot(x = 'datetime', y = 'Latitude') 
df.plot(x = 'datetime', y = 'Longitude') 
df.describe()
df['Year'] = df['datetimestr'].str.slice(0,4)
df['Month'] = df['datetimestr'].str.slice(5,7)
df.groupby('Year').describe()
df.groupby('Month').describe()