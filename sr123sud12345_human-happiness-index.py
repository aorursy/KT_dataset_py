from pandas import Series,DataFrame

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")

import statistics
i=1
col=['Dance', 'Shopping', 'Fun with friends', "Parents' advice",

       'Eating to survive', 'Pets', 'Darkness', 'Fear of public speaking',

       'Economy Management', 'Healthy eating', 'Decision making',

       'Workaholism', 'Friends versus money', 'Loneliness', 'God', 'Dreams',

       'Number of friends', 'Socializing', 'Entertainment spending', 'height','weight',

       'age', 'gender', 'alcohol', 'smoke', 'only_child', 'area', 'internet',

       'music', 'movie', 'education']

df=DataFrame(columns=col)

#a1=pd.Series(np.arange(0,29,1),index=col)

#a1=pd.Series(a,index=col)
col1=["name","comments"]

#a2=pd.Series(c,index=col1)

df1=DataFrame(columns=col1)
#personal_details

c1=input("plz enter your name ",)

c2=input("significant_comment(hostel,room_number,etc) ",)

b22=int(input("your age ",))

b20=int(input("your Height in centimeter ",))

b21=int(input("your Weight ",))

b23=(input("your Gender ",))

b26=(input("Are you the only child of the family ",))

b27=(input("which one you preffer Urban or Rural  ",))

b31=input("please enter your eduction(pick one statemenet from above)  ")
b29=int(input("your liking for music "))

b30=int(input("your liking  for movies "))

b1=int(input("your level of intrest in dance ",))

b2=int(input("your liking for shopping ",))

b3=int(input("your level of Fun with friends ",))

b4=int(input("How often do you listen to youe parents advice ",))

b5=int(input("How much focused towadrs the goal of life you are  ",))

b6=int(input("your affection towars pet ",))

b7=int(input("your level fear with darkness ",))

b8=int(input("your level of Fear in public speaking ",))

b25=int(input("Level of smoker you are (on a scale of 1 to 3) ",))

b24=int(input("Level of Alcholic you are (on a scale of 1 to 3) ",))

b9=int(input("Rate yourself on sacle of Economy managment  ",))

b10=int(input("Rate youeself on a scale of healty eating ",))

b11=int(input("How much good you are in Decision making ",))

b12=int(input("Level workaholic you are ",))

b13=int(input(" which one is more important Friends versus money (0 for friend 5 for money) ",))

b14=int(input("Level of Loneliness you are in  ",))

b15=int(input("Your belief in GOD ",))

b16=int(input("How often do you see dreams ",))

b17=int(input(" number of friends (5 for a lot and 0 for least)",))

b18=int(input("How much do you socialize with others ",))

b19=int(input("Level of your spending on Entertainment ",))

#b24=int(input("your age",))

#b25=int(input("your Height ",))

#b26=int(input(" your Weight",))

#b28=int(input("your Gender ",))

#b29=int(input("Are you the only child of the family",))

#b30=int(input("which one you preffer Urban or Rural ",))

b28=int(input("Your level of internet uses(on a scale of 1 to 3) ",))
a=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31]
#col1=["name","comments"]

c=[c1,c2]

a2=pd.Series(c,index=col1)

#df1=DataFrame(columns=col1)

df1=df1.append(a2,ignore_index=True)

df1
len(col)==len(a)
#df=DataFrame(columns=col)

#a1=pd.Series(np.arange(0,29,1),index=col)

a1=pd.Series(a,index=col)

df=df.append(a1,ignore_index=True)

df
df_merge=df

df_merge["name"]=c1

df_merge["comments"]=c2
i=i+1

i
df.to_csv("mp_data.csv")
#df_merge.to_csv("mp_merge.csv")