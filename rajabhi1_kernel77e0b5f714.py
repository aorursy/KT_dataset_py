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
file_path="../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv"

df_initial=pd.read_csv(file_path,encoding='latin1')

df_initial.head(20)
col=df_initial.columns

df_initial['Median Income']

#for changing the (X) value

df_initial=df_initial.replace('(X)',-1)

df_initial[df_initial=='(X)'].count()       

        

        

          

        



df_initial=df_initial.replace('-',-1)

df_initial[df_initial['Median Income']=='2500-'] .count()
l=[]

for i in df_initial['Median Income']:

    i=str(i)

    if(',' in i ):

        l.append(250000)

    elif(i == 'nan')   :

        l.append(0)

    else:

        l.append(int(i)) 

print(l)              
df_initial.drop('Median Income', axis = 1, inplace = True)

df_initial['Median Income']=l

df_initial=df_initial.replace(-1,48000)

df_initial=df_initial.replace(0,48000)    
df_initial.head()
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(16,6))

sns.kdeplot(data=df_initial['Median Income'],shade=True)