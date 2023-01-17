# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

FILE="../input/accident.csv"

df=pd.read_csv(FILE)

print(df)

# Any results you write to the current directory are saved as output.


states = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 

          6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 

          11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 

          16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 

          21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 

          25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 

          28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 

          32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 

          36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 

          40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 43: 'Puerto Rico', 

          44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 

          48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 52: 'Virgin Islands', 

          53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'}



print(wdf.head(4))
#we are going to create another column..

df['state']=df['STATE'].apply(lambda x:states[x])

drunkdf=df.groupby(['state']).agg({'DRUNK_DR':sum})

#print(drunkdf.head(2))

totalcount=df['state'].value_counts().to_frame()

#print(totalcount.head(2))

drunkdf['total']=totalcount['state']

drunkdf['per']=drunkdf['DRUNK_DR']/drunkdf['total']

drunkdf['per']=drunkdf['per']*100

drunkdf=drunkdf.sort(columns=['per'], ascending=[0])

print(drunkdf.head(4))
import matplotlib.pyplot as plt

import seaborn as sns

#displaying only top 20 drunk drivers states

drunkdf=drunkdf.head(20)

sns.barplot(x=drunkdf.per,y=drunkdf.index,orient='h')

plt.xlabel('percentage of drunk drivers over total drivers')

plt.ylabel('state')

plt.title('Top 20 most drunk drivers states')