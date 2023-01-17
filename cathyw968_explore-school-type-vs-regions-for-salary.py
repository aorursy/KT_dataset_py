# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import pandas as pd

import re

import matplotlib

from matplotlib import pyplot as plt



df1 = pd.read_csv("../input/salaries-by-college-type.csv")

df2 = pd.read_csv("../input/salaries-by-region.csv")





#Combine the college data

cols = ['School Name','Region']

college_combined = pd.merge(left=df1, right=df2[cols], how='inner',on='School Name')

college_combined.head()
#rename the columns and clean the dollar culumns

college_combined.columns=["Name","Type","str_50","mid_50","mid_10","mid_25","mid_75","mid_90","Region"]

#clean the dollar columns

dollar_col= ["str_50","mid_50","mid_10","mid_25","mid_75","mid_90"]

for x in dollar_col:

    college_combined[x] = college_combined[x].str.replace("$", "")

    college_combined[x] = college_combined[x].str.replace(",", "")

    college_combined[x] = pd.to_numeric(college_combined[x])

college_combined.head()
# compare the Starting Median Salary and the Mid-Career Median Salary, we can see the increase for each school type vs region

# type engineering vs region

engineering = college_combined[college_combined["Type"]=="Engineering"]

engineering1 = engineering[["str_50", "mid_50", "Region"]]

engineering2= engineering1.groupby("Region").str_50.mean().to_frame()  #get the mean by region

engineering3= engineering1.groupby("Region").mid_50.mean().to_frame()

#change to int

e = [int(i) for i in engineering2["str_50"]]

engineering2["str_50"]= e

e11 = [int(i) for i in engineering3["mid_50"]]

engineering3["mid_50"] =e11

# put start 50% and mid 50% together

type_engineering= engineering2.assign(mid_50 = engineering3.mid_50.values)

type_engineering= type_engineering.reset_index()

type_engineering
# type state vs region

state = college_combined[college_combined["Type"]=="State"]

state1 =state[["str_50", "mid_50", "Region"]]

state2= state1.groupby("Region").str_50.mean().to_frame()  #get the mean by region

state3= state1.groupby("Region").mid_50.mean().to_frame()

#change to int

s = [int(i) for i in state2["str_50"]]

state2["str_50"]= s

s11 = [int(i) for i in state3["mid_50"]]

state3["mid_50"] =s11

# put start 50% and mid 50% together

type_state= state2.assign(mid_50 = state3.mid_50.values)

type_state= type_state.reset_index()



# type party vs region

party = college_combined[college_combined["Type"]=="Party"]

party1 =party[["str_50", "mid_50", "Region"]]

party2= party1.groupby("Region").str_50.mean().to_frame()  #get the mean by region

party3= party1.groupby("Region").mid_50.mean().to_frame()

#change to int

p = [int(i) for i in party2["str_50"]]

party2["str_50"]= p

p11 = [int(i) for i in party3["mid_50"]]

party3["mid_50"] =p11

# put start 50% and mid 50% together

type_party= party2.assign(mid_50 = party3.mid_50.values)

type_party= type_party.reset_index()





# type Liberal Arts vs region

arts = college_combined[college_combined["Type"]=="Liberal Arts"]

art1 = arts[["str_50", "mid_50", "Region"]]

art2= art1.groupby("Region").str_50.mean().to_frame()  #get the mean by region

art3= art1.groupby("Region").mid_50.mean().to_frame()

#change to int

a = [int(i) for i in art2["str_50"]]

art2["str_50"]= a

a11 = [int(i) for i in art3["mid_50"]]

art3["mid_50"] =a11

# put start 50% and mid 50% together

type_art= art2.assign(mid_50 = art3.mid_50.values)

type_art= type_art.reset_index()





# type Liberal Arts vs region

ivy = college_combined[college_combined["Type"]=="Ivy League"]

ivy1 = ivy[["str_50", "mid_50", "Region"]]

ivy2= ivy1.groupby("Region").str_50.mean().to_frame()  #get the mean by region

ivy3= ivy1.groupby("Region").mid_50.mean().to_frame()

#change to int

i = [int(i) for i in ivy2["str_50"]]

ivy2["str_50"]= i

i11 = [int(i) for i in ivy3["mid_50"]]

ivy3["mid_50"] =i11

# put start 50% and mid 50% together

type_ivy= ivy2.assign(mid_50 = ivy3.mid_50.values)

type_ivy= type_ivy.reset_index()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

fig = plt.figure(figsize=(8,5))





matplotlib.rc('grid', alpha = .5, color = '#e3dfdf')   #color the grid lines

matplotlib.rc('axes', edgecolor = '#67746A')           #color the graph edge



#for school type engineering

x = type_engineering["str_50"]

y = len(type_engineering.index) - type_engineering.index + 1

labels = type_engineering['Region']

plt.yticks(y, labels)

plt.scatter(x, y, color='r', label = 'Engineering (50% people)')

x1 = type_engineering['mid_50']

plt.scatter(x1, y, color='r')



for i in range(len(type_engineering.index)):

    plt.plot([x[i], x1[i]], [y[i], y[i]], color='red')

    



#for school type state

x2 = type_state["str_50"]

#y = len(type_state.index) - type_state.index + 1

#labels = type_state['Region']

#plt.yticks(y, labels)

plt.scatter(x2, y, color='orange', label = 'State (50% people)')

x3 = type_state['mid_50']

plt.scatter(x3, y, color='orange')



for i in range(len(type_state.index)):

    plt.plot([x2[i], x3[i]], [y[i], y[i]], color='orange')



#for school type party

x4 = type_party["str_50"]

plt.scatter(x4, y, color='green', label = 'Party (50% people)')

x5 = type_party['mid_50']

plt.scatter(x5, y, color='green')



for i in range(len(type_party.index)):

    plt.plot([x4[i], x5[i]], [y[i], y[i]], color='green')

    



#for school type liberty arts

x6 = type_art["str_50"]

plt.scatter(x6, y, color='purple', label = 'Liberal Arts (50% people)')

x7 = type_art['mid_50']

plt.scatter(x7, y, color='purple')



for i in range(len(type_party.index)):

    plt.plot([x6[i], x7[i]], [y[i], y[i]], color='purple')



    

#for school type Ivy League

y = [4]                             #y has index for 5 area, Ivy League is only in Northeast

x8 = type_ivy["str_50"]

plt.scatter(x8, y, color='black', label = 'Ivy League (50% people)')

x9 = type_ivy['mid_50']

plt.scatter(x9, y, color='black')



plt.plot([x8[0], x9[0]], [y[0], y[0]], color='black')  #only one region: northeast



    

plt.xlabel('US $  from Starting Median Salary to Mid-Career Median Salary')

plt.ylabel('Area')

plt.title('Salary Information (Region vs School type)')

plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98))



plt.grid(True) #turn grid on



plt.show()