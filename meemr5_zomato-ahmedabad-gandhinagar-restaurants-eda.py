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
data = pd.read_csv('../input/zomato_restaurants_in_India.csv')
# We only want data of Ahmedabad & Gandhinagar

data = data[(data.city=="Ahmedabad") | (data.city=="Gandhinagar")]

data.shape
data.head()
# Checking for redundant data

data["res_id"].nunique()
data.drop_duplicates(["res_id"],keep='first',inplace=True)

data.shape
data.set_index("res_id",inplace=True)
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.countplot(x='city',data=data)

plt.title("Restaurants count")
#plt.figure(figsize=(35,15))

#plt.figure(figsize=(8,8))

sns.barplot(y=(data[data["city"]=="Ahmedabad"]["name"].value_counts()).index[:10],x=(data[data["city"]=="Ahmedabad"]["name"].value_counts()).values[:10])

plt.title("Top 10 Restaurants of Ahmedabad",fontsize=10)
sns.barplot(y=(data[data["city"]=="Gandhinagar"]["name"].value_counts()).index[:10],x=(data[data["city"]=="Gandhinagar"]["name"].value_counts()).values[:10])

plt.title("Top 10 Restaurants of Gandhinagar",fontsize=10)
Ahmedabad = data[data["city"]=="Ahmedabad"]

Gandhinagar = data[data["city"]=="Gandhinagar"]



plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.barplot(y=Ahmedabad["establishment"].value_counts().index,x=Ahmedabad["establishment"].value_counts().values)

plt.title("Establishment Counts (Ahmedabad)")



plt.subplot(1,2,2)

sns.barplot(y=Gandhinagar["establishment"].value_counts().index,x=Gandhinagar["establishment"].value_counts().values)

plt.title("Establishment Counts (Gandhinagar)")
# We only want the area (Ex: not "SBR Social, Bodakdev" but Bodakdev)

test = Ahmedabad.copy()

for i in test.index:

    test.loc[i,"locality"] = str(test.loc[i,"locality"]).split(', ')[-1] if str(test.loc[i,"locality"]).split(', ')[-1] != 'Gandhinagar' else str(test.loc[i,"locality"]).split(', ')[-2]

test["locality"].value_counts().index
Ahmedabad = test.copy()

#Gandhinagar_new = FetchArea(Gandhinagar).copy()
test = Gandhinagar.copy()

for i in test.index:

    test.loc[i,"locality"] = str(test.loc[i,"locality"]).split(', ')[-1] if str(test.loc[i,"locality"]).split(', ')[-1] != 'Gandhinagar' else str(test.loc[i,"locality"]).split(', ')[-2]

test["locality"].value_counts().index
Gandhinagar = test.copy()
plt.figure(figsize=(30,20))

plt.subplot(1,2,1)

sns.barplot(x=Ahmedabad['locality'].value_counts().values,y=Ahmedabad['locality'].value_counts().index)

plt.title("#Restaurants in each Area (Ahmedabad)")



plt.subplot(1,2,2)

sns.barplot(x=Gandhinagar['locality'].value_counts().values,y=Gandhinagar['locality'].value_counts().index)

plt.title("#Restaurants in each Area (Gandhinagar)")
Ahmedabad['cuisines'].value_counts()[:10]
# To see which cuisines are higher in counts, we will make a map to count values

test = Ahmedabad.copy()

Cuisines_Count = {}

for cuisines in test['cuisines']:

    for c in str(cuisines).split(', '):

        if c in Cuisines_Count:

            Cuisines_Count[c] = Cuisines_Count[c] + 1

        else:

            Cuisines_Count[c] = 1
sortedC = sorted(Cuisines_Count.items(),key=lambda kv:kv[1])[::-1]

import collections

Cuisines_Count = collections.OrderedDict(sortedC)

# Reference : https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
Cuisines_Count_A = Cuisines_Count

plt.figure(figsize=(7,7))

sns.barplot(y=[str(x) for x in Cuisines_Count.keys()][:10],x=[int(x) for x in Cuisines_Count.values()][:10])

plt.title("10 most comman Cuisines in Ahmedabad")
test = Gandhinagar.copy()

Cuisines_Count = {}

for cuisines in test['cuisines']:

    for c in str(cuisines).split(', '):

        if c in Cuisines_Count:

            Cuisines_Count[c] = Cuisines_Count[c] + 1

        else:

            Cuisines_Count[c] = 1

            

sortedC = sorted(Cuisines_Count.items(),key=lambda kv:kv[1])[::-1]

import collections

Cuisines_Count = collections.OrderedDict(sortedC)



Cuisines_Count_G = Cuisines_Count

plt.figure(figsize=(7,7))

sns.barplot(y=[str(x) for x in Cuisines_Count.keys()][:10],x=[int(x) for x in Cuisines_Count.values()][:10])

plt.title("10 most comman Cuisines in Gandhinagar")
data["average_cost_for_two"].describe()
ind = data[data["average_cost_for_two"]==0].index.values

lis = []

for i in ind:

    if [data.loc[i,"establishment"],data.loc[i,"locality"]] not in lis:

        lis.append([data.loc[i,"establishment"],data.loc[i,"locality"]])

lis
import math

avg_cost = []

for [e,l] in lis:

    length = data[(data["establishment"]==e) & (data["locality"]==l)].shape[0]

    avg_cost.append(math.ceil(data[(data["establishment"]==e) & (data["locality"]==l) & (data["average_cost_for_two"]!=0)]["average_cost_for_two"].mean()*length/(length-1)))

avg_cost
d = {}

for ([i,j],c) in zip(lis,avg_cost):

    d[i] = c

d
for i in data[data["average_cost_for_two"]==0].index.values:

    est = data.loc[i,"establishment"]

    data.loc[i,"average_cost_for_two"] = d[est]
plt.figure(figsize=(16,8))

plt.subplot(1,2,1)

sns.distplot(data[data["city"]=="Gandhinagar"]["average_cost_for_two"],rug=True)

plt.title("Average cost for 2 (Gandhinagar)")



plt.subplot(1,2,2)

sns.distplot(data[data["city"]=="Ahmedabad"]["average_cost_for_two"],rug=True)

plt.title("Average cost for 2 (Ahmedabad)")
#ahm = Ahmedabad.copy()

plt.figure(figsize=(25,10))

plt.subplot(1,2,1)

sns.barplot(y="establishment",x="average_cost_for_two",data=Ahmedabad)

plt.title("Average cost for two per Establishment (Ahmedabad)")



#gndh = data[data["city"]=="Gandhinagar"]

plt.subplot(1,2,2)

sns.barplot(y="establishment",x="average_cost_for_two",data=Gandhinagar)

plt.title("Average cost for two per Establishment (Gandhinagar)")
# Finding Locality wise average cost in Ahmedabad

localityXAvgCost_A = {}

for loc in Ahmedabad["locality"].unique():

    localityXAvgCost_A[loc] = Ahmedabad[Ahmedabad["locality"]==loc]["average_cost_for_two"].mean()

# Sorting Dictionary

sorted_LA_A = sorted(localityXAvgCost_A.items(),key=lambda kv:kv[1])[::-1]

localityXAvgCost_A = collections.OrderedDict(sorted_LA_A)



plt.figure(figsize=(15,15))

sns.barplot(y=[str(x) for x in localityXAvgCost_A.keys()],x=[int(x) for x in localityXAvgCost_A.values()])

plt.title("Average cost per Locality (Ahmedabad)")
#plt.figure(figsize=(8,8))

sns.barplot(y="locality",x="average_cost_for_two",data=Gandhinagar)

plt.title("Average cost per Locality (Gandhinagar)")
# Plots top 20(default) highlights of a City

def Highlights(data,city,top=20):    

    highlight_Counts = {}

    for highlights in data["highlights"]:

        for highlight in str(highlights).split(', '):

            if highlight[0]=='[':

                highlight = highlight[1:]

            if highlight[-1]==']':

                highlight = highlight[:-1]



            if highlight in highlight_Counts:

                highlight_Counts[highlight] += 1

            else:

                highlight_Counts[highlight] = 1



    sorted_highlights_a = sorted(highlight_Counts.items(),key= lambda x : x[1])[::-1]

    highlight_Counts = collections.OrderedDict(sorted_highlights_a)



    plt.figure(figsize=(10,10))

    sns.barplot(y=[str(x) for x in highlight_Counts.keys()][:top],x=[int(x) for x in highlight_Counts.values()][:top])

    plt.title("Top " +str(top) + " Highlights of Restaurants in " + city)

    

    return highlight_Counts
highlights_Ahmedabad = Highlights(Ahmedabad,"Ahmedabad")
highlights_Gandhinagar = Highlights(Gandhinagar,"Gandhinagar")
total_res_A = Ahmedabad.shape[0]

Digital_Payments_A = 0 



for highlights in Ahmedabad["highlights"]:

    if "'Digital Payments Accepted'" in str(highlights).split(', '):

        Digital_Payments_A +=1

    elif "'Credit Card'" in str(highlights).split(', '):

        Digital_Payments_A +=1

    elif "'Debit Card'" in str(highlights).split(', '):

        Digital_Payments_A +=1

    elif "'Sodexo'" in str(highlights).split(', '):

        Digital_Payments_A +=1





# Data to plot

labels = ["Yes","No"]

sizes = [Digital_Payments_A,total_res_A-Digital_Payments_A]



# Plot

plt.figure(figsize=(8,8))

plt.pie(sizes,labels=labels,startangle=90,autopct='%.1f%%',colors=["red","yellow"],wedgeprops={ 'linewidth' : 3,'edgecolor' : "black" })

plt.title("Digital Payment (Ahmedabad)")
total_res_G = Gandhinagar.shape[0]

Digital_Payments_G = 0 



for highlights in Gandhinagar["highlights"]:

    if "'Digital Payments Accepted'" in str(highlights).split(', '):

        Digital_Payments_G +=1

    elif "'Credit Card'" in str(highlights).split(', '):

        Digital_Payments_G +=1

    elif "'Debit Card'" in str(highlights).split(', '):

        Digital_Payments_G +=1

    elif "'Sodexo'" in str(highlights).split(', '):

        Digital_Payments_G +=1





# Data to plot

labels = ["Yes","No"]

sizes = [Digital_Payments_G,total_res_G-Digital_Payments_G]



# Plot

plt.figure(figsize=(8,8))

plt.pie(sizes,labels=labels,shadow=True,startangle=90,autopct='%.1f%%',wedgeprops={ 'linewidth' : 3,'edgecolor' : "black" })

plt.title("Digital Payment (Gandhinagar)")
plt.figure(figsize=(15,7))

plt.subplot(1,2,1)

plt.pie([highlights_Ahmedabad["'Delivery'"],total_res_A-highlights_Ahmedabad["'Delivery'"]],

       startangle=90,autopct='%.1f%%',

       labels=["Yes","No"])

plt.title("Delivery (Ahmedabad)")



plt.subplot(1,2,2)

plt.pie([highlights_Gandhinagar["'Delivery'"],total_res_G-highlights_Gandhinagar["'Delivery'"]],

       startangle=90,autopct='%.1f%%',

       labels=["Yes","No"])

plt.title("Delivery (Gandhinagar)")
# Ahmedabad & Gandhinar ratings distributions

plt.figure(figsize=(20,40))



plt.subplot(4,2,1)

sns.distplot(Ahmedabad["aggregate_rating"])

plt.title("Ahmedabad aggregate ratings")



plt.subplot(4,2,2)

sns.distplot(Gandhinagar["aggregate_rating"])

plt.title("Gandhinagar aggregate ratings")



plt.subplot(4,2,3)

sns.countplot(Ahmedabad["rating_text"])

plt.title("Ahmedabad rating text counts")

plt.xticks(rotation=45)



plt.subplot(4,2,4)

sns.countplot(Gandhinagar["rating_text"])

plt.title("Gandhinagar rating text counts")



plt.subplot(4,2,5)

sns.distplot(Ahmedabad["votes"])

plt.title("Ahmedabad #votes distribution")



plt.subplot(4,2,6)

sns.distplot(Gandhinagar["votes"])

plt.title("Gandhinagar #votes distribution")



plt.subplot(4,2,7)

sns.distplot(Ahmedabad["photo_count"])

plt.title("Ahmedabad #Photos distribution")



plt.subplot(4,2,8)

sns.distplot(Gandhinagar["photo_count"])

plt.title("Gandhinagar #Photos distribution")
def BestRestaurants(city,esta):

    data = city[city["establishment"]==esta]["name"].value_counts()[:10]

    ratings = {}

    #print(data.index.values)

    for name in data.index.values:

        ratings[name] = city[(city["establishment"]==esta) & (city["name"]==name)]["aggregate_rating"].mean()

    #print(ratings)

    sorted_ratings = sorted(ratings.items(),key=lambda x:x[1])[::-1]

    ratings = collections.OrderedDict(sorted_ratings)

    #print(ratings)

    plt.figure(figsize=(8,8))

    sns.barplot(y=[str(x) for x in ratings.keys()],x=[float(x) for x in ratings.values()])

    plt.title("Top 10 " + esta + " Restaurants in " + str(city["city"].values[0]))

    plt.xlabel("Average Ratings")

    plt.ylabel("Restaurant Names")
BestRestaurants(Ahmedabad,"['Quick Bites']")
BestRestaurants(Ahmedabad,"['Casual Dining']")
BestRestaurants(Ahmedabad,"['Café']")
BestRestaurants(Ahmedabad,"['Dessert Parlour']")
BestRestaurants(Gandhinagar,"['Quick Bites']")
BestRestaurants(Gandhinagar,"['Food Court']")
BestRestaurants(Gandhinagar,"['Sweet Shop']")
from statistics import mean

def BestRestaurantsLocal(city,locality,cuisine):

    data = city[city["locality"]==locality]

    ratings = {}

    #print(data.index.values)

    for i in data.index.values:

        if cuisine in str(data.loc[i,"cuisines"]).split(', '):

            if data.loc[i,"name"] not in ratings:

                ratings[data.loc[i,"name"]] = [data.loc[i,"aggregate_rating"]]

            else:

                ratings[data.loc[i,"name"]].append(data.loc[i,"aggregate_rating"])

    for r in ratings.keys():

        ratings[str(r)] = mean(ratings[str(r)])

    sorted_ratings = sorted(ratings.items(),key=lambda x:x[1])[::-1]

    ratings = collections.OrderedDict(sorted_ratings)

    #print(ratings)

    plt.figure(figsize=(8,8))

    if len(ratings)>10:

        sns.barplot(y=[str(x) for x in ratings.keys()][:10],x=[float(x) for x in ratings.values()][:10])

        #plt.title("Top 10 " + cuisine + " Restaurants in " + locality)

        #plt.xlabel("Average Ratings")

        #plt.ylabel("Restaurant Names")

    else:

        sns.barplot(y=[str(x) for x in ratings.keys()],x=[float(x) for x in ratings.values()])

    plt.title("Top 10 " + cuisine + " Restaurants in " + locality)

    plt.xlabel("Average Ratings")

    plt.ylabel("Restaurant Names")
BestRestaurantsLocal(Ahmedabad,'Bodakdev','North Indian')
BestRestaurantsLocal(Ahmedabad,'Satellite','Fast Food')