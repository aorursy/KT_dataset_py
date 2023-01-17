# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import os
print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/results.csv")
#read_csv command opens the file in the "../input/results.csv" directory
data.info()
#The info() command in the results.csv shows properties such as column names, types, and number of data.
data.head()
#the head() command shows the first 5 elements in the results.csv
away_team_won=data[data.away_score>data.home_score]
home_team_won=data[data.home_score>data.away_score]
draw=data[data.home_score==data.away_score]
#Created dataframe separately for away won ,home won and draw .conditions
y=np.array([away_team_won.shape[0],home_team_won.shape[0],draw.shape[0]])
#make a list of first elements of the dimensions of the generated data frame
x=["awayteamwon","hometeamwon","draw"]
plt.bar(x,y)
plt.title("result of football worldwide")
plt.show()
#list was shown using the plt.bar property         
turkeyh=data[data.home_team=='Turkey']
turkeya=data[data.away_team=='Turkey']

ta_away=turkeya[turkeya.away_score>turkeya.home_score]
ta_home=turkeya[turkeya.home_score>turkeya.away_score]
ta_draw=turkeya[turkeya.home_score==turkeya.away_score]

th_away=turkeyh[turkeyh.away_score>turkeyh.home_score]
th_home=turkeyh[turkeyh.home_score>turkeyh.away_score]
th_draw=turkeyh[turkeyh.home_score==turkeyh.away_score]

t1=np.array([th_away.shape[0],th_home.shape[0],th_draw.shape[0]])
t2=np.array([ta_home.shape[0],ta_away.shape[0],ta_draw.shape[0]])

z=["opponentwon","turkeywon","draw"]
plt.subplot(1,2,1,)
plt.bar(z,t1, width=0.2)
plt.title("Turkey's home statistics")
plt.subplot(1,2,2)

plt.bar(z,t2,width=0.2)
plt.title("Turkey's away statistics")
plt.show()
#here home and away turkey statistics shown
             
             
             
             
i=0
list=[]
while (i<len(data)-1):
    list=list+[data.loc[:,"date"][i][0:4]]
    i=i+1
#took the first 4 digits of the date in data and create new list
#A new list with the name list has been created.

plt.hist(list,bins=100)
plt.xlabel("frekans")
plt.ylabel("football matches")
plt.title("Histogram")
plt.show()
#histogram of the list was drawn
frame=pd.DataFrame({'year':list})
years=frame.year.unique()
print(years)
len(years)
#list by converting to dataframe and find unique
i=0
a=1
list1=[]

while i <len(list)-1:
    if list[i]==list[i+1]:
        a=a+1
        i=i+1

    else:
        list1=list1+[a]
        a=1
        i=i+1
list1=list1+[a]
print(list1)
len(list1)
#how many times a year were repeated
plt.plot(years,list1)
plt.xlabel("Years")
plt.ylabel("Football Matches")
plt.show()
#we can do the work done by the histogram in the last two lines.
goal01=0
goal23=0
goal46=0
goal7over=0
i=0

while i<len(data)-1:
    if (data.home_score[i] +data.away_score[i] in [0,1]):
        goal01=goal01+1
        
    elif (data.home_score[i] +data.away_score[i] in [2,3]):
        goal23=goal23+1
        
    elif (data.home_score[i] +data.away_score[i] in [4,5,6]):
        goal46=goal46+1 
        
    elif (data.home_score[i] +data.away_score[i]>7):
        goal7over=goal7over+1       
    i=i+1
y=[goal01,goal23,goal46,goal7over]
x=["goal01","goal23","goal46","goal7over"]
plt.bar(x,y)
plt.show()
#how many matches 0-1,2-3,4-6,7+ finished
