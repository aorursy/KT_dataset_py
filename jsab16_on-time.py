# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sql
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


df_train['time'].describe()
time = df_train['time']
time = time % (24*60)#*60#*60*10

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

df_train['place_id'].value_counts().head(10) #get the top places to breakout time
offset=0 # This can be adjusted if we figure out what time midnight is
time = df_train[df_train['place_id']==8772469670]['time']

timeToTest=24*60#*60#*60*10

time = (time+offset) % timeToTest

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
time = df_train[df_train['place_id']==1623394281]['time']
time = (time+offset) % timeToTest

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
time = df_train[df_train['place_id']==1308450003]['time']
time = (time+offset) % timeToTest

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
time = df_train[df_train['place_id']==4823777529]['time']
time = (time+offset) % timeToTest

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#Strong case for this dataset being in minutes.
#Let's see how much time this data has been collected for
print('Time since start of data collection: ' + str(round(df_train['time'].max()/(24*60*365.25),2)) + ' Years.')
