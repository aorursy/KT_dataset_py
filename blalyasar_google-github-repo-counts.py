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
import requests
from bs4 import BeautifulSoup
import re 
import pandas as pd
df = pd.read_csv("/kaggle/input/googlegithubrepositoryanalsysis/google-github-analiz.csv")
df.head()

df = df.drop(columns=["Unnamed: 0"])

df = df.drop(columns=["Unnamed: 0.1"])
df.head()
df.describe()
# Programlama dili olarak python
# Lisans olarak Apache-2.0
# eksik veri yok
df.isnull().sum()
# Kullanılan tüm diller
df['Dil'].unique()
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
df['Dil'].value_counts().plot.bar()
plt.figure(figsize=(15,12))
df['Dil'].value_counts().plot.pie(autopct='%9.0f%%')
plt.xlabel(" ",fontsize = 20)
plt.ylabel(" ", fontsize = 20)
plt.title("Google Github Repository Programlama Dilleri")
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
df['Lisans'].value_counts().plot.bar()
