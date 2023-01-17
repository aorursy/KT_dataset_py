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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=sns.load_dataset("tips")

df.head()
#1.relplot(kind='scatter') when not mentioned

sns.relplot(x='total_bill',y='tip',data=df)  
# 2.For 3 variables, add hue

sns.relplot(x='total_bill',y='tip',hue='sex',data=df)
#3.For 4 variables, add hue + style

sns.relplot(x='total_bill',y='tip',hue='sex',style='time',data=df)
#4. when numeric data type is used in hue it show one color & their intensity to represent the value

sns.relplot(x='total_bill',y='tip',hue='size',data=df)
#5.if you are more of bigger the better type ...then

sns.relplot(x='total_bill',y='tip',hue='sex',size='size',data=df)
#6.well the circles arent big enough so to buff it up..

sns.relplot(x='total_bill',y='tip',hue='sex',size='size',sizes=((15, 200)),data=df) # 15 & 200 can be replaced with min and max values too
#7.to see tip vs total_bill for male and female individual.add col

sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',data=df)
#8.to see tip vs total_bill for male smoker and non smoker and female smoker and non smoker,add row

sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',row='smoker',data=df)
#9.UI lets make them thinner by reducing the aspect

sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',aspect=.4,data=df)
#10.well each fig got thinner but all stackup on eachother.lets wrap them up using col_wrap

#sns.relplot(x='total_bill',y='tip',hue='smoker',col='sex',row='size',aspect=.4,col_wrap=5,data=df)

#row and col_wrap cannot be used..Noted
#11.see we can plot like this But can't see more of any pattern

sns.relplot(x="day", y="total_bill", data=df)
#12.But this looks visually appealing

sns.catplot(x="day", y="total_bill",kind='strip', data=df) #no need to mention strip as it is the default
#13. Lets try 'Swarm'

sns.catplot(x="day", y="total_bill",kind='swarm', data=df) 

#although it looks same to striplot it doesn't overlap datas.so we can see the distribution more clearly in swarm but using it large data set might look aweful.
#14.Nothing fancy just added an another variable 'sex' using hue.

sns.catplot(x="day", y="total_bill",hue='sex',kind='swarm', data=df) 
#15.what if i want to reverse the order of days,add order which accepts list..

sns.catplot(x="day", y="total_bill",hue='sex',kind='strip',order=["Sun","Thur","Fri","Sat"],data=df)
#16. we can filterout certain category here i kicked out persons whose team size=3. using query function ..but do know it is a dataframe function

sns.catplot(x="size", y="total_bill",kind='swarm',data=df.query('size!=3'))
#17.we'll try box plot to see the distribution of values

sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='box')
#18.to stack up hue value use dodge=false

sns.catplot(x="day", y="total_bill",hue='sex',dodge=False,data=df,kind='box')
#19.boxen plots is similar to box but it shows more distribution of data in larger data set then box plots

sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='boxen')
sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin')
#20 .Violin plots looks good but sometimes when u have negative values then u need to cut extreme negative ends using 'cut'

sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',bw=.10,cut=10)
#20 .to save visual space both values in hue can be shown together using 'split'

sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',split=True,bw=.10,cut=10)
#20 .instead of the box plot inside you can see the actual distribution using 'inner'

sns.catplot(x="day", y="total_bill",hue='sex',data=df,kind='violin',split=True,inner='sticks')
#21.u can combine swarm plot with violin plots too..



violinPlot = sns.catplot(x="day", y="total_bill", kind="violin", inner='sticks', data=df) # put inner =None as it looks terrible  with swarms already

sns.catplot(x="day", y="total_bill",color="k",kind='swarm', data=df,ax=violinPlot.ax)  #add color to differtiate from background
#22.the most common plot used all over..

sns.catplot(x="day", y="size",data=df,kind='bar')
#22.it is used to estimate the number of values it contains itself ..meaning:how many cats ,dogs,donkeys present in a animals column assuming only these 3 animals exist in it

sns.catplot(x='day',data=df,kind='count')
sns.catplot(x="size", y="tip",hue='sex',data=df,kind='point')