# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the data

delay1 = pd.read_csv("/kaggle/input/usa-flights-2018/01_18.csv")

delay2 = pd.read_csv("/kaggle/input/usa-flight-2018/02_18.csv")

delay3 = pd.read_csv("/kaggle/input/usa-flight-2018/03_18.csv")

delay4 = pd.read_csv("/kaggle/input/usa-flight-2018/04_18.csv")

delay5 = pd.read_csv("/kaggle/input/usa-flight-2018/05_18.csv")

delay6 = pd.read_csv("/kaggle/input/usa-flight-2018/06_18.csv")

delay = pd.concat([delay1, delay2, delay3, delay4, delay5, delay6], axis=0)

delay.head()
# Create delay Column

delay.loc[:,"OTP"] = 0 #ontime performance



#filter

filter_Delay = (delay["DEP_DEL15"] == 1) | (delay["ARR_DEL15"] == 1) 

filter_DnA = (delay["DEP_DEL15"] == 1) & (delay["ARR_DEL15"] == 0)



# apply delay filter to DEL15

delay.loc[filter_Delay, "OTP"] = 1
# Function to create graph

def plotDelay(x, data, ax, xlab, hue=None, xticklabs=None, top=0):

    sns.countplot(x=x, data=data, ax=ax, hue=hue)

    total = len(data)

    #annotation

    ymax = 0

    for p in ax.patches:

        ax.text(p.get_x()+p.get_width()/2, (p.get_height() - top), int(p.get_height()), color="w", fontweight="ultralight", ha="center", va="center", alpha=0.3)

        ax.text(p.get_x()+p.get_width()/2, p.get_height()/2, "{0} %".format(round((p.get_height() * 100) / total), 2), color="w", fontweight="bold", ha="center", va="center")

        if ymax < p.get_height():

            ymax = p.get_height()



    #settings

    ax.tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

    ax.set_yticks([]) #show only the extreme value

    if(xticklabs != None):

        ax.set_xticklabels(xticklabs) #rename the xticklabels

    ax.set_ylabel("") #hiding the "count" label

    ax.set_xlabel(xlab) #x label

    sns.despine(left=True, bottom=True) #hiding axes left/right/top/bottom
sns.set(style="white")



fig, ax = plt.subplots(1,3, figsize=(22,12))



plotDelay(x="DEP_DEL15", data=delay, ax=ax[0], top=130000, 

          xlab="Number of Departures Delays", xticklabs=["Not Delayed","Delayed"])

plotDelay(x="ARR_DEL15", data=delay, ax=ax[1], top=130000, 

          xlab="Number of Arrivals Delays", xticklabs=["Not Delayed","Delayed"])

plotDelay(x="OTP", data=delay, ax=ax[2], top=130000, 

          xlab="On-Time Performance (OTP)", xticklabs=["Not Delayed","Delayed"])
# The on-time arrival performance knowing that departure has been delayed

round((len(delay[filter_DnA]) * 100) / len(delay[delay["DEP_DEL15"] == 1]), 2)
delay_by_days = delay.groupby(by="DAY_OF_WEEK")["ARR_DEL15"].agg(["sum", "count"])

delay_by_days.loc[:,"percent"] = (delay_by_days["sum"] * 100) / delay_by_days["count"]
fig, ax = plt.subplots(2,1, figsize=(22,15))

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

sns.barplot(x=delay_by_days.index, y=delay_by_days["percent"], ax=ax[0])

sns.countplot(x=delay["DAY_OF_WEEK"], ax=ax[1])

# Annotations

for p in ax[0].patches:

        ax[0].text(p.get_x()+p.get_width()/2, (p.get_height()/2), "{0} %".format(round(p.get_height(),1)), 

                color="w", fontweight="bold", ha="center", va="center", alpha=0.9)

ax[0].tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax[0].set_yticks([]) #show only the extreme value

ax[0].set_xticklabels(days)

ax[0].set_ylabel("") #hiding the "count" label

ax[0].set_xlabel("") #x label

ax[0].set_title("Percentage of Delays")



for p in ax[1].patches:

        ax[1].text(p.get_x()+p.get_width()/2, (p.get_height()/2), int(p.get_height()), 

                color="w", fontweight="bold", ha="center", va="center", alpha=0.9)

ax[1].tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax[1].set_yticks([]) #show only the extreme value

ax[1].set_xticklabels(days)

ax[1].set_ylabel("") #hiding the "count" label

ax[1].set_xlabel("") #x label

ax[1].set_title("Number of Flights")



sns.despine(left=True, bottom=True)

plt.tight_layout()



fig.savefig("DayOfWeek_delays.png", transparent=True, bbox_inches='tight', pad_inches=0)
delay_by_dmonth = delay.groupby(by="DAY_OF_MONTH")["ARR_DEL15"].agg(["sum", "count"])

delay_by_dmonth.loc[:,"percent"] = (delay_by_dmonth["sum"] * 100) / delay_by_dmonth["count"]


fig, ax = plt.subplots(1,1, figsize=(17,9))

sns.barplot(x=delay_by_dmonth.index, y=delay_by_dmonth["percent"])



# Annotations

for p in ax.patches:

        ax.text(p.get_x()+p.get_width()/2, (p.get_height()/2), "{0} %".format(round(p.get_height(),1)), 

                color="w", fontweight="bold", ha="center", va="center", alpha=1, rotation="vertical")

ax.tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax.set_yticks([]) #show only the extreme value

ax.set_ylabel("") #hiding the "count" label

ax.set_xlabel("Days of Month") #x label

ax.set_title("Percentage of Delays")

sns.despine(left=True, bottom=True)
def getHr(x):

    x = str(x)

    if len(x) == 4:

        return int(x[:2])

    if len(x) == 3:

        return int(x[:1])

    if len(x) < 3:

        return int(0)

    

delay["HR_DEP_TIME"] = delay["CRS_DEP_TIME"].apply(getHr)
delay_by_hour = delay.groupby(by="HR_DEP_TIME")["ARR_DEL15"].agg(["sum", "count"])

delay_by_hour.loc[:,"percent"] = (delay_by_hour["sum"] * 100) / delay_by_hour["count"]
fig, ax = plt.subplots(1,1, figsize=(22,12))

order=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4]

order_labels=["05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", 

              "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", 

              "23:00", "00:00", "01:00", "02:00", "03:00", "04:00"]

sns.barplot(x=delay_by_hour.index, y=delay_by_hour["percent"], order=order)

# Annotations

for p in ax.patches:

        ax.text(p.get_x()+p.get_width()/2, (p.get_height()/2), "{0} %".format(round(p.get_height(),1)), 

                color="w", fontweight="bold", ha="center", va="center", alpha=1, rotation="vertical")

ax.tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax.set_yticks([]) #show only the extreme value

ax.set_ylabel("") #hiding the "count" label

ax.set_xlabel("Hours Of The Day") #x label

ax.set_xticklabels(order_labels)

ax.set_title("Percentage of Delays")

sns.despine(left=True, bottom=True)



fig.savefig("HourOfDay_delays.png", transparent=True, bbox_inches='tight', pad_inches=0)
delay_by_month = delay.groupby(by="MONTH")["ARR_DEL15"].agg(["sum", "count"])

delay_by_month.loc[:,"percent"] = (delay_by_month["sum"] * 100) / delay_by_month["count"]
fig, ax = plt.subplots(1,1, figsize=(16,10))

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',

         'September', 'October', 'November', 'December']

sns.barplot(x=delay_by_month.index, y=delay_by_month["percent"], ax=ax)



# Annotations

for p in ax.patches:

        ax.text(p.get_x()+p.get_width()/2, (p.get_height()/2), "{0} %".format(round(p.get_height(),1)), 

                color="w", fontweight="bold", ha="center", va="center", alpha=0.9)

ax.tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax.set_yticks([]) #show only the extreme value

ax.set_xticklabels(month)

ax.set_ylabel("") #hiding the "count" label

ax.set_xlabel("") #x label

ax.set_title("Percentage of Delays")



sns.despine(left=True, bottom=True)

plt.tight_layout()



fig.savefig("Month_delays.png", transparent=True, bbox_inches='tight', pad_inches=0)

delay_by_md = pd.pivot_table(delay, index="DAY_OF_WEEK", columns="MONTH", values="ARR_DEL15", 

                             aggfunc=lambda x: (np.sum(x) * 100) / len(x))



fig, ax = plt.subplots(1,1, figsize=(22,10))

sns.heatmap(delay_by_md, annot=True, xticklabels=month, yticklabels=days)

ax.set_ylabel("") #hiding the "count" label

ax.set_xlabel("") #x label

ax.set_title("Percentage of Delays") #title

plt.yticks(rotation=0)
dist_delay = delay.groupby("DISTANCE_GROUP")["ARR_DEL15"].agg(lambda x: (np.sum(x) * 100) / len(x))
fig, ax = plt.subplots(1,1, figsize=(22,10))

sns.barplot(x=dist_delay.index, y=dist_delay.values)

# Annotations

for p in ax.patches:

        ax.text(p.get_x()+p.get_width()/2, (p.get_height()/2), "{0} %".format(round(p.get_height(),1)), 

                color="w", fontweight="bold", ha="center", va="center", alpha=0.9)

ax.tick_params(bottom="off", top="off", left="off", right="off") #Hiding Tick Marks

ax.set_yticks([]) #show only the extreme value

ax.set_ylabel("") #hiding the "count" label

ax.set_xlabel("") #x label

ax.set_title("Percentage of Delays") #title

sns.despine(left=True, bottom=True)