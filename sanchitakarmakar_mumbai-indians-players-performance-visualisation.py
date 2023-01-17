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
# Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing the dataset

# In the given excel file we have 20 separate sheets for different stadiums.

# Lets import each sheet one by one



# Bangalore Stadium

df_bang = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Bengaluru')
df_bang.head()
# Displaying column names

# Notice that Ave is renamed as Ave.1 since there were two columns with same name. Similarly for Runs.1, Player Name.1 and SR.1

df_bang.columns
# Delhi Stadium

df_delhi = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Delhi')

df_delhi.head()
# Jaipur Stadium

df_jpr = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Jaipur')

df_jpr.head()
# Hyderabad Stadium

df_hyd = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Hyderabad')

df_hyd.head()
# Mohali Stadium

df_moh = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Mohali')

df_moh.head()
# Mumbai Stadium

df_mum = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Mumbai')

df_mum.head()
# Kolkata Stadium

df_kol = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Kolkata')

df_kol.head()
# Chennai Stadium

df_chn = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Chennai')

df_chn.head()
# Ahmedabad Stadium

df_amd = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Ahmedabad')

df_amd.head()
# Cuttak Stadium

df_ctk = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Cuttak')

df_ctk.head()
# Nagpur Stadium

df_ng = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Nagpur')

df_ng.head()
# Dharamshala Stadium

df_dh = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Dharamshala')

df_dh.head()
# Kochi Stadium

df_kch = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Kochi')

df_kch.head()
# Indore Stadium

df_ind = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Indore')

df_ind.head()
# Visakhapatnam Stadium

df_vis = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Visakhapatnam')

df_vis.head()
# Pune Stadium

df_pn = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Pune')

df_pn.head()
# Raipur Stadium

df_rpr = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Raipur')

df_rpr.head()
# Ranchi Stadium

df_rnh = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Ranchi')

df_rnh.head()
# Kanpur Stadium

df_knp = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Kanpur')

df_knp.head()
# Rajkot Stadium

df_rjk = pd.read_excel('../input/mumbai-indians-stadium-wise-performance/UPTADED PERFORMANCE IN STADIUM LIST 2019.xlsx','Rajkot')

df_rjk.head()
# Now lets create a list containing all the datasets for easy access in visualisation coming up

mylist = [df_bang, df_delhi, df_jpr, df_hyd, df_moh, df_mum, df_kol, df_chn,df_amd, df_ctk, df_ng, df_dh, df_kch, df_ind,df_vis,df_pn,df_rpr,df_rnh,df_knp, df_rjk]
# Creating a list for the names of all stadiums they are situated at

mylist2 = ['Bangalore','Delhi','Jaipur','Hyderabad','Mohali','Mumbai','Kolkata','Chennai','Ahmedabad','Cuttak','Nagpur','Dharamshala','Kochi','Indore','Visakhapatnam','Pune','Raipur','Ranchi','Kanpur','Rajkot']
# Creating a list of the names of all players

Players  = ['Rohit Sharma','Yuvraj Singh','Hardik Pandiya','Quinton de kock','Anukul Roy','Ishan Kishan','Kieron Pollard',

'Suryakumar Yadav','Krunal Pandya','Lashith Malinga','Jasprit bumrah','Mitchell Mcclenagham','Rahul Chahar','Jayant Yadav',

'Ben cutting','Siddhesh Lad','Beuran Hendricks','Aditya Tare','Mayank Markande','Alzarri Joseph','Evin Lewis','Rasikh Salam',

'Barinder Saran','Jason Behrendorff','Anmolpreet Singh','Pankaj Jaswal']
# Now lets begin Visualizations !!!
# Lets see Batting Average of different players for Bangalore Stadium.

from matplotlib import rcParams

rcParams['figure.figsize'] = 10,7



b = sns.barplot(x='Player Name', y = 'Ave', data = mylist[0] )



plt.rcParams["xtick.labelsize"] = 10

plt.xticks(rotation = 90)

plt.tight_layout()

plt.xlabel('Player Name',fontsize=15)

plt.ylabel('Average',fontsize=15)

plt.title('Bangalore', fontsize = 15)



for p in b.patches:

    b.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points')

plt.show()
# What if we create visualization for batting average of all the players at different stadiums all at once!

# Excited ?
# So lets GO for it!

f, axes = plt.subplots(10,2, figsize=(15,80), sharex = False, sharey = True) 

plt.subplots_adjust(bottom=0.2, wspace=None, hspace=1.2)

i=0

mylist3 = []

while i < 20:

    for j in range(0,10):

        for k in range(0,2):

            b = sns.barplot(x='Player Name', y = 'Ave', data = mylist[i], ax=axes[j][k])

            mylist3.append(axes[j][k])

            

            axes[j][k].set_xticklabels(Players, rotation = 90, fontsize=10)

            axes[j][k].set_ylabel('Batting Average',fontsize=10)

            axes[j][k].title.set_text(mylist2[i])



            for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

            i = i+1

plt.show()
# Similarly lets create visualization for batting Strike Rate for all the players at different stadiums
f, axes = plt.subplots(10,2, figsize=(15,80), sharex = False, sharey = True) 

plt.subplots_adjust(bottom=0.2, wspace=None, hspace=1.2)

i=0

mylist3 = []

while i < 20:

    for j in range(0,10):

        for k in range(0,2):

            b = sns.barplot(x='Player Name', y = 'SR', data = mylist[i], ax=axes[j][k])

            mylist3.append(axes[j][k])

            

            axes[j][k].set_xticklabels(Players, rotation = 90, fontsize=10)

            axes[j][k].set_ylabel('Strike Rate(Batting)',fontsize=10)

            axes[j][k].title.set_text(mylist2[i])



            for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

            i = i+1

plt.show()
# Similarly lets create visualization for bowling Average for all the players at different stadiums
f, axes = plt.subplots(10,2, figsize=(15,80), sharex = False, sharey = True) 

plt.subplots_adjust(bottom=0.2, wspace=None, hspace=1.2)

i=0

mylist3 = []

while i < 20:

    for j in range(0,10):

        for k in range(0,2):

            b = sns.barplot(x='Player Name', y = 'Ave.1', data = mylist[i], ax=axes[j][k])

            mylist3.append(axes[j][k]) 

            axes[j][k].set_xticklabels(Players, rotation = 90, fontsize=10)

            axes[j][k].set_ylabel('Bowling Average',fontsize=10)

            axes[j][k].title.set_text(mylist2[i])



            for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

            i = i+1

plt.show()
# Similarly lets create visualization for bowling Economy for all the players at different stadiums

f, axes = plt.subplots(10,2, figsize=(15,80), sharex = False, sharey = True) 

plt.subplots_adjust(bottom=0.2, wspace=None, hspace=1.2)

i=0

mylist3 = []

while i < 20:

    for j in range(0,10):

        for k in range(0,2):

            b = sns.barplot(x='Player Name', y = 'Econ', data = mylist[i], ax=axes[j][k])

            mylist3.append(axes[j][k]) 

            axes[j][k].set_xticklabels(Players, rotation = 90, fontsize=10)

            axes[j][k].set_ylabel('Economy',fontsize=10)

            axes[j][k].title.set_text(mylist2[i])



            for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

            i = i+1

plt.show()
# Similarly lets create visualization for bowling Strike Rate for all the players at different stadiums

f, axes = plt.subplots(10,2, figsize=(15,80), sharex = False, sharey = True) 

plt.subplots_adjust(bottom=0.2, wspace=None, hspace=1.2)

i=0

mylist3 = []

while i < 20:

    for j in range(0,10):

        for k in range(0,2):

            b = sns.barplot(x='Player Name', y = 'SR.1', data = mylist[i], ax=axes[j][k])

            mylist3.append(axes[j][k]) 

            axes[j][k].set_xticklabels(Players, rotation = 90, fontsize=10)

            axes[j][k].set_ylabel('Strike Rate (Bowling)',fontsize=10)

            axes[j][k].title.set_text(mylist2[i])



            for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

            i = i+1

plt.show()
# From above graphs we can easily get insights of how players performance at given stadium 
# Now lets visualize the performance of a given player at different stadiums
# Visualising the Batting Average of each player at different stadiums

plt.style.use('dark_background')

rcParams['figure.figsize'] = 8,3

for j in range(0,26):

    mylist4=[]

    for i in range(0,20):

        mylist4.append(mylist[i].at[j,'Ave'])

    #print(mylist4)

    b = sns.barplot(x = mylist2, y = mylist4)

    plt.xticks(rotation = 90)

    plt.title(Players[j])

    plt.ylabel('Batting Average')

    plt.ylim(0,70)

    for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

    plt.show()
# Visualising the Batting Strike Rate of each player at different stadiums

for j in range(0,26):

    mylist4=[]

    for i in range(0,20):

        mylist4.append(mylist[i].at[j,'SR'])

    #print(mylist4)

    b = sns.barplot(x = mylist2, y = mylist4)

    plt.xticks(rotation = 90)

    plt.title(Players[j])

    plt.ylabel('Strike Rate(Batting)')

    for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

    plt.show()
for j in range(0,26):

    mylist4=[]

    for i in range(0,20):

        mylist4.append(mylist[i].at[j,'Econ'])

    #print(mylist4)

    b = sns.barplot(x = mylist2, y = mylist4)

    plt.xticks(rotation = 90)

    plt.title(Players[j])

    plt.ylim(0,17)

    plt.ylabel('Bowling Economy')

    for p in b.patches:

                b.annotate(format(p.get_height(), '.1f'), 

                (p.get_x() + p.get_width() / 2., p.get_height()), 

                ha = 'center', va = 'center', 

                xytext = (0, 9), 

                textcoords = 'offset points')

    plt.show()
# Similarly we can try for bowling Strike Rate, Average, etc