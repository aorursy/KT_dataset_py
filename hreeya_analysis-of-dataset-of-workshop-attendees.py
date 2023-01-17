# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path="../input/all delhi database.xlsx"

df=pd.read_excel(path)
df.head()
df.columns=["Name","Email_Id","Mobile_No.","College"]
df.head()
df.tail()
df.columns
df.describe(include = "all")
df.shape
df['College'].value_counts()
# use the inline backend to generate the plots within the browser

%matplotlib inline 



import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('ggplot') # optional: for ggplot-like style



# check for latest version of Matplotlib

print ('Matplotlib version: ', mpl.__version__) # >= 2.0.
df['College'].value_counts().plot(kind='bar')
df['College'].value_counts().plot(kind='pie',figsize=(24, 28),

                            autopct='%1.1f%%', # add in percentages

                            startangle=90,     # start angle 90° (Africa)

                            shadow=True,    )   # add shadow 