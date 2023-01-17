# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19327/COVID-19-3.27.top30.xlsx')

df.head()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covid19327/COVID-19-3.27-top30(ex-china)65day.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['China'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Orange')

    plt.xlabel(col)

    plt.ylabel('China')

    plt.tight_layout()

    plt.show()
df.plot.area(y=['country','China','Italy','Spain', 'US', 'Japan'],alpha=0.4,figsize=(12, 6));
df1 = pd.read_excel('/kaggle/input/covid19327/COVID-19-3.27-top30-500.xlsx')

df1.head()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/covid19327/COVID-19-3.27-500.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)