# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data['Date']= pd.to_datetime(data['Date'])

data["month"] = data["Date"].dt.month

data = data.drop("Unnamed: 0",axis=1)

data.tail()
volume = data.groupby("type").agg("sum")["Total Volume"]

fig = plt.figure()

langs = ['conventional','organic']

plt.pie(volume, labels = langs,autopct='%1.2f%%')

plt.title("Total Volume")

plt.show()
def autolabel(rects,ax):

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%.2E' % float(height),ha='center', va='bottom')

volume = data.groupby(["year","type"]).agg("sum")["Total Volume"]

volume = pd.DataFrame(volume)

plt.figure(figsize=(8,6))

bar1 =plt.bar(["2015","2016","2017","2018"], volume.loc[pd.IndexSlice[:, 'organic'], "Total Volume"], width = 0.3, color='red',label="organic")

bar2= plt.bar(["2015","2016","2017","2018"],volume.loc[(slice(None), slice('conventional')), :]["Total Volume"],color='lightblue',

        width = 0.3,bottom=volume.loc[pd.IndexSlice[:, 'organic'], "Total Volume"],label="conventional")

autolabel(bar1,plt.gca())

autolabel(bar2,plt.gca())

plt.gca().axes.get_yaxis().set_visible(False)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.title("Total Volume")

plt.legend()

plt.show()
volume = data.groupby(["month","type"]).agg("sum")["Total Volume"]

months = [1,2,3,4,5,6,7,8,9,10,11,12]

volume = pd.DataFrame(volume)

plt.figure(figsize=(20,6))

bar1 =plt.bar(months, volume.loc[pd.IndexSlice[:, 'organic'], "Total Volume"], width = 0.3, color='red',label="organic")

bar2= plt.bar(months,volume.loc[(slice(None), slice('conventional')), :]["Total Volume"],color='lightblue',

        width = 0.3,bottom=volume.loc[pd.IndexSlice[:, 'organic'], "Total Volume"],label="conventional")

autolabel(bar1,plt.gca())

autolabel(bar2,plt.gca())

plt.gca().axes.get_yaxis().set_visible(False)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.title("Total Volume")

plt.gca().set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

plt.gca().set_xticklabels(['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec','Dec'])

plt.legend()

plt.show()
pricebymonth = data.groupby(["year","month","type"]).agg({'AveragePrice': "mean", 'Total Volume':"sum"})

m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec','Dec']

ticks = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(20,15))

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2015, :,"conventional"],"Total Volume"]

         ,label="2015 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2015, :,"organic"],"Total Volume"]

         ,label="2015 organic")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2016, :,"conventional"],"Total Volume"]

         ,label="2016 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2016, :,"organic"],"Total Volume"]

         ,label="2016 organic")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2017, :,"conventional"],"Total Volume"]

         ,label="2017 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2017, :,"organic"],"Total Volume"]

         ,label="2017 organic")

plt.plot([x+1 for x in range(3)],pricebymonth.loc[pd.IndexSlice[2018, :,"conventional"],"Total Volume"]

         ,label="2018 conventional")

plt.plot([x+1 for x in range(3)],pricebymonth.loc[pd.IndexSlice[2018, :,"organic"],"Total Volume"]

         ,label="2018 organic")

ax = plt.gca()

ax.set_xticks(ticks)

ax.set_xticklabels(m)

ax.xaxis.grid(True,alpha=0.4)

ax.yaxis.grid(True,alpha=0.4)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.legend()

plt.title("Total Volume")

plt.show()
pricebymonth = data.groupby(["month","type"]).agg({'AveragePrice': "mean", 'Total Volume':"sum"})

m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec','Dec']

ticks = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(10,8))

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[ :,"conventional"],"Total Volume"]

         ,label="conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[ :,"organic"],"Total Volume"]

         ,label="organic")

ax = plt.gca()

ax.fill_between([x+1 for x in range(12)], 

                       pricebymonth.loc[pd.IndexSlice[ :,"conventional"],"Total Volume"], pricebymonth.loc[pd.IndexSlice[ :,"organic"],"Total Volume"], 

                       facecolor='grey', 

                       alpha=0.2)

ax.set_xticks(ticks)

ax.set_xticklabels(m)

ax.xaxis.grid(True,alpha=0.4)

ax.yaxis.grid(True,alpha=0.4)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

#ax.yaxis.set_ticks(np.arange(0.9, 1.9, 0.05))

plt.legend()

plt.title("Overall total Volume")

plt.show()
average = data.groupby(["year","region"]).agg("mean")["AveragePrice"]

average = pd.DataFrame(average)
def autolabel(rects,h,ax):

    for rect in rects:

        height = rect.get_height()

        if(height >h):

            rect.set_color("blue")

        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,'%0.2f' % float(height),ha='center', va='bottom')

def getnames(year):

    names = list()

    for index in average.loc[pd.IndexSlice[year, :],:].index:

        names.append(index[1])

    return names
fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, 1,figsize=(26, 10), sharex=True, sharey=True)

axs = [ax1,ax2,ax3,ax4]

year = 2015

for ax in axs:   

    bar = ax.bar(getnames(year),average.loc[pd.IndexSlice[year, :],"AveragePrice"],color="lightgrey",alpha=0.8)

    autolabel(bar,2,ax)

    ax.tick_params(axis ='x', rotation = 90)

    ax.axes.get_yaxis().set_visible(False)

    for spine in ax.spines.values():

        spine.set_visible(False)

    ax.set_title(str(year)+" average price",y = 0.95)

    year = year +1

plt.show()
pricebymonth = data.groupby(["year","month","type"]).agg({'AveragePrice': "mean", 'Total Volume':"sum"})
m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec','Dec']

ticks = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(20,15))

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2015, :,"conventional"],"AveragePrice"]

         ,label="2015 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2015, :,"organic"],"AveragePrice"]

         ,label="2015 organic")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2016, :,"conventional"],"AveragePrice"]

         ,label="2016 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2016, :,"organic"],"AveragePrice"]

         ,label="2016 organic")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2017, :,"conventional"],"AveragePrice"]

         ,label="2017 conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[2017, :,"organic"],"AveragePrice"]

         ,label="2017 organic")

plt.plot([x+1 for x in range(3)],pricebymonth.loc[pd.IndexSlice[2018, :,"conventional"],"AveragePrice"]

         ,label="2018 conventional")

plt.plot([x+1 for x in range(3)],pricebymonth.loc[pd.IndexSlice[2018, :,"organic"],"AveragePrice"]

         ,label="2018 organic")

ax = plt.gca()

ax.set_xticks(ticks)

ax.set_xticklabels(m)

ax.xaxis.grid(True,alpha=0.4)

ax.yaxis.grid(True,alpha=0.4)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

ax.yaxis.set_ticks(np.arange(0.9, 2.2, 0.05))

plt.legend()

plt.title("Average Price")

plt.show()
pricebymonth = data.groupby(["month","type"]).agg({'AveragePrice': "mean", 'Total Volume':"sum"})
m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec','Dec']

ticks = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(10,8))

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[ :,"conventional"],"AveragePrice"]

         ,label="conventional")

plt.plot([x+1 for x in range(12)],pricebymonth.loc[pd.IndexSlice[ :,"organic"],"AveragePrice"]

         ,label="organic")

ax = plt.gca()

ax.fill_between([x+1 for x in range(12)], 

                       pricebymonth.loc[pd.IndexSlice[ :,"conventional"],"AveragePrice"], pricebymonth.loc[pd.IndexSlice[ :,"organic"],"AveragePrice"], 

                       facecolor='grey', 

                       alpha=0.2)

ax.set_xticks(ticks)

ax.set_xticklabels(m)

ax.xaxis.grid(True,alpha=0.4)

ax.yaxis.grid(True,alpha=0.4)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

ax.yaxis.set_ticks(np.arange(0.9, 1.9, 0.05))

plt.legend()

plt.title("Overall average price")

plt.show()
ax=data["AveragePrice"].plot.kde(figsize=(10,5),title="Average price density function")

for spine in ax.spines.values():

    spine.set_visible(False)

ax.xaxis.set_ticks(np.arange(0, 3.4, 0.2))

ax.yaxis.set_ticks(np.arange(0, 1.2, 0.2))

ax.set_xlim(0 , 3.2)

ax.yaxis.grid(True,alpha=0.4)

ax.xaxis.grid(True,alpha=0.4)
plt.figure(figsize=(10,5))

box = plt.boxplot([data.loc[data["type"]=="organic","AveragePrice"], data.loc[data["type"]=="conventional","AveragePrice"] ] , patch_artist=True)

ax = plt.gca()

ax.set_xticklabels(('organic','conventional'))

for spine in ax.spines.values():

    spine.set_visible(False)

ax.yaxis.grid(True,alpha=0.4)

plt.setp(box["boxes"], facecolor="lightgrey")

plt.title("Average Price")

plt.show()
fig, (ax1,ax2)  = plt.subplots(1, 2,figsize=(20,10) ,sharex=True,sharey=True)



ax1.hist(data.loc[(data["type"]=="organic"),"AveragePrice"])

ax1.set_title("organic, average price")

for spine in ax1.spines.values():

    spine.set_visible(False)

ax1.xaxis.grid(False)



ax2.hist(data.loc[(data["type"]=="conventional"),"AveragePrice"])

ax2.set_title("conventional, average price")

for spine in ax2.spines.values():

    spine.set_visible(False)

ax2.xaxis.grid(False)
fig, (ax1,ax2)  = plt.subplots(2, 1,figsize=(20,20) ,sharex=False,sharey=False)

m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec']

ax1.boxplot([data.loc[(data["type"]=="organic") & (data["month"]==1),"AveragePrice"],

             data.loc[(data["type"]=="organic") & (data["month"]==2),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==3),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==4),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==5),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==6),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==7),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==8),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==9),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==10),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==11),"AveragePrice"],

            data.loc[(data["type"]=="organic") & (data["month"]==12),"AveragePrice"]] , patch_artist=True)

ax1.set_xticklabels(m)

ax1.yaxis.set_ticks(np.arange(0.4, 3.3, 0.1))

for label in ax1.yaxis.get_ticklabels()[::2]:

    label.set_visible(False)

for spine in ax1.spines.values():

    spine.set_visible(False)

ax1.yaxis.grid(True,alpha=0.4)

ax1.title.set_text('average prices for organic')



ax2.boxplot([data.loc[(data["type"]=="conventional") & (data["month"]==1),"AveragePrice"],

             data.loc[(data["type"]=="conventional") & (data["month"]==2),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==3),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==4),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==5),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==6),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==7),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==8),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==9),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==10),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==11),"AveragePrice"],

            data.loc[(data["type"]=="conventional") & (data["month"]==12),"AveragePrice"]] , patch_artist=True)

ax2.set_xticklabels(m)

for spine in ax2.spines.values():

    spine.set_visible(False)

ax2.yaxis.grid(True,alpha=0.4)

ax2.yaxis.set_ticks(np.arange(0.4, 2.5, 0.1))

for label in ax2.yaxis.get_ticklabels()[::2]:

    label.set_visible(False)

ax2.title.set_text('average prices for conventional')
plt.figure(figsize=(10,20))

sns.set(style="whitegrid")

ax = sns.boxplot(x ="AveragePrice",y ="region" ,data=data[data["type"]=="organic"], orient="h", palette="Set2").set_title('Organic')
plt.figure(figsize=(10,20))

sns.set(style="whitegrid")

ax = sns.boxplot(x ="AveragePrice",y ="region" ,data=data[data["type"]=="conventional"], orient="h", palette="Set2").set_title('conventional')
plt.figure(figsize=(15,15))

plt.scatter(data.loc[data["type"]=='conventional',"Total Volume"],data.loc[data["type"]=='conventional',"AveragePrice"]

            ,alpha=1,color='blue',label="conventional")

plt.scatter(data.loc[data["type"]=='organic',"Total Volume"],data.loc[data["type"]=='organic',"AveragePrice"]

            ,alpha=1,color='red',label="organic")

ax = plt.gca()

ax.xaxis.grid(True,alpha=0.4)

ax.yaxis.grid(True,alpha=0.4)

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.legend()

plt.xlabel("Total Volume")

plt.ylabel("Average Price")

plt.show()