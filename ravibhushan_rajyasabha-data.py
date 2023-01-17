# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Rajyasabha_QnA.csv')
df.info()
df.head()
df['answer_date']=pd.to_datetime(df['answer_date'])
months=['Jan','Feb','Mar','April','May','June','July','Aug','Sept','Oct','Nov','Dec']

plt.figure(figsize=(10,10))

ax=sns.countplot(x=df['answer_date'].dt.month,hue=df['answer_date'].dt.year,data=df)

plt.xticks(rotation=90)

plt.title('Monthly question asked')

ax.set(xticklabels=months)

for p in ax.patches:

    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.1,p.get_y()+p.get_height()+130),rotation=90)

plt.show()
plt.figure(figsize=(20,20))

ax=sns.countplot(x=df['ministry'],data=df)

plt.xticks(rotation=90)

plt.title("Total question asked by each ministry")

#ax.set(xticklabels=months)

for p in ax.patches:

    ax.annotate('{:.0f}'.format(p.get_height()),(p.get_x()+0.1,p.get_y()+p.get_height()+30),rotation=90)

plt.show()
dx=df.groupby('ministry')

print(dx.head().to_string())