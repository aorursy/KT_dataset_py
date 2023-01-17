import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import requests



response = requests.get('http://lib.stat.cmu.edu/datasets/boston')

data = response.text

for i, line in enumerate(data.split('\n')):

    if i<24:

        print(f'{i}   {line}' )

    elif i>1020:

        print(f'{i}   {line}' )
# enbale below two lines to view entire text data

# for i, line in enumerate(data.split('\n')):

#     print(f'{i}   {line}' )
columns = []

for i,line in enumerate(data.split('\n')[:-1]):

    if i > 6 and i <21:

        if re.match('^\s*([A-Z]+)',line):

            columns.append(re.match('^\s*([A-Z]+)',line).groups()[0])

columns
l = {}

for i,line in enumerate(data.split('\n')[:-1]):

    if i>21:

        x = re.findall('[0-9.]+',line)



        if len(x)>3:

            l[i]=x

        else:

            l[i-1].extend(x)
# visualizing the data in dictionary l

for key in list(l.keys())[:3]:

    print(l[key])
df=pd.DataFrame(columns=columns)

df.columns=columns

for i in l:

    row = l[i]

    df.loc[i,:]=row

df.reset_index(drop=True,inplace=True)

df