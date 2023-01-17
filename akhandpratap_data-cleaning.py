import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import Libraries

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Load Data

df=pd.read_excel('/kaggle/input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx')
df.head()
#Working on Highest percentile column

df['Highest percentile'].head()
new=df['Highest percentile'].str.split('\n',n=3,expand=True)

new.head()

new
new1=new[1].str.split('/',n=2,expand=True)

new1
#HP as Highest percentile column subdivided into 3 columns 

df['HP_Percentile']=new[0]

df['HP_Frequency'] = new1[1]

df['HP_Subject'] = new[2]

df.head()
df.info()
df.drop('Highest percentile',axis=1,inplace=True)

df.head()