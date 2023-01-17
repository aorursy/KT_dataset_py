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
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
#I am choosing a dataset that looks at the favorability of democratic candidates in the 2020 race. I'm interested in this 

#data because I would like to start making some conclusions about the presidential race based off my own read of the data.

#I know that this is a small step in that direction, but it's nice to at least get moving in that direction. It also helps that,

#other people have done work with this dataset and I can at least have a template to shoot for.
polls = pd.read_csv("../input/favorability_polls_rv_2019.csv")
df=polls #making polls a dataframe

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots() #honestly, I googled this and found that this is how to create multiple line charts on top ofe each other for each candidate

df.groupby('politician').plot(y='favorable',x='end_date',ax=ax, legend=False) #I am grouping the favorable measures by politician over time. 

#My next step is to try and only plot favorability for one candidate at a time.
x=-1 #this represents my original idea of how to look at each candidate. Setting x as a pseudo index here

for row in df.itertuples(): #iterate through the dataframe by row and call out each "cell" as a value equal to the column header

    x=x+1  #adding a value to x every iteration through

    for candidate in row: #iterating through each row of data looking for candidate information

        if candidate=='Michael F. Bennet': #if the candidate variable equals a certain name...

            print(x)#then print x. I was going to add x to dictionary where I could eventually reference the row that x is saying contains a candidate. This seemed...complex. So I switched strategies.   

            
cand='Michael Bloomberg'#setting up the cand variable to ingest the candidate information

is_candidate =  df['politician']==cand #creating a variable that tests if the politician column in df is equal to the cand variable I've already set

df_cand = df[is_candidate] #creating a new dataset that only looks for columns where the is_candidate is true

df_cand.plot(y='favorable',x='end_date') #plotting the favoribality of Mike Bloomberg in this case. This is much easier!