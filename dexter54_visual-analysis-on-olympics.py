import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
regions=pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")

athletes=pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
#First we will check athletes dataframe

athletes.head()
athletes.info()

# We have 271116 athletes in total
athletes.columns

# We have each athlete's ID, Name, Sex, Age, Heigh, Weight, Team, NOC, Games, Year, Season, City, Sport, Event, Medal as features
athletes.describe()
regions.head()

#We are seeing that NOC is common for two dataframes so we can use it to merge there dataframes
regions.info() #In total, we have 22y region
regions.describe()
data=pd.concat([athletes,regions],axis=1)
data.head(10) #Now we now long version of NOC
data.corr() #.corr() shows correlations between features in data. To see it clearly, we will use heatmap
sns.heatmap(data.corr(),annot=True,fmt=".1f")

# There is one obvious correlation which is between height and weight. 
# I wonder is variety of games increased by the time. I will use horizantal barplot for this.

new_data=data[["Year","Sport"]]

years=data["Year"].unique()

sports=[]

for y in years:

    sports.append(len(data[data["Year"]==y]["Sport"].unique()))

sports_and_year=pd.DataFrame({"Year":years,"Number of Sport":sports})

sorted_data=sports_and_year.sort_values(by="Year",ascending=True).reset_index(drop=True)



#Visualizing

sorted_data.plot(kind="barh",x="Year",y="Number of Sport",figsize=(15,15),cmap="viridis",legend=False)

plt.xlabel("Number of Sports")

plt.title("Variety of Sports with Respect to Year")

plt.show()

# We can see there is an dramatic increate till 1992 after that there is an olympic competiton in every two year
year_sex=data[["Year","Sex"]]



sex_counter={"Year":[],"Gender":[],"Number":[]}

for i in years:

    males=year_sex[(year_sex["Year"]==i)&(year_sex["Sex"]=="M")].Sex.count()

    sex_counter["Gender"].append("Male")

    sex_counter["Number"].append(males)

    sex_counter["Year"].append(i)

    females=year_sex[(year_sex["Year"]==i)&(year_sex["Sex"]=="F")].Sex.count()

    sex_counter["Gender"].append("Female")

    sex_counter["Number"].append(females)

    sex_counter["Year"].append(i)

sex_data=pd.DataFrame(sex_counter)





#Visualizing Data



plt.figure(figsize=(20,20))

sns.barplot(data=sex_data,x="Year",y="Number",hue="Gender")

plt.legend(fontsize=20,loc="upper left")



plt.xlabel("Year",color="c",size=15)

plt.ylabel("Number",color="c",size=15)

plt.xticks(rotation=90)

plt.grid()

plt.title("Number of Athletes in each Year with respect to Sex",color="c",size=15)

plt.show()



#We can see that number of females are increased till 1992. After that,olympics are made in every 2 years.

#In 1994-1998-2002-2006-2010-2014 olympics, there is a clear trend that shows increase in number of females.In 1992-1996-2000-2004-2008-1012-2012 olympics which are more crowded, number of females is mostly increased.

    

    
#Now take a look to ratio between number of males and females in each year



#Preparing data

years=list(sex_data["Year"].unique())

ratio=[]

for i in years:

    male_num=int(sex_data[(sex_data["Year"]==i)&(sex_data["Gender"]=="Male")]["Number"])

    female_num=int(sex_data[(sex_data["Year"]==i)&(sex_data["Gender"]=="Female")]["Number"])

    ratio.append(female_num/male_num)

ratio_df=pd.DataFrame(list(zip(years,ratio)),columns=["Years","Ratio"]).sort_values(by="Years",ascending=True).reset_index(drop=True)



#Visualization

fig, ax = plt.subplots(figsize=(15,15))

sns.barplot(data=ratio_df,x="Years",y="Ratio",palette="Oranges",ax=ax)



plt.title("Ratio between males and females by year",color="g",size=20)

plt.xlabel("Ratios",color="g",size=15)

plt.ylabel("Years",color="g",size=15)

plt.xticks(rotation=90)

plt.yticks(np.arange(0,1.1,step=0.1))

plt.legend()



#We can see an increase in ratio between females and males


#Preparing Data

data["Medal"]=data["Medal"].fillna(value=0,axis=0)#Filling Nan values in medals as 0

new_data=data[["Year","Team","Medal"]]

new_data["Medal"]=[1 if (i=="Gold" or i=="Silver" or i=="Bronze" )else 0 for i in new_data["Medal"]]



date=[]

teams=[]

medals=[]

for i in years:

    countries=list(new_data[new_data["Year"]==i]["Team"].unique())

    for c in  countries:

        total=new_data[(new_data["Team"]==c)&(new_data["Year"]==i)].Medal.sum()

        medals.append(total)

        date.append(i)

        teams.append(c)
#Visualisation by plotly

medals_countries=pd.DataFrame(zip(date,teams,medals),columns=["Year","Team","Medals"])

medals_countries.drop(medals_countries[medals_countries["Medals"]==0].index,inplace=True,axis=0)#Dropping countries which have 0 medals

medals_countries.sort_values(by=["Year","Medals"],inplace=True,axis=0,ascending=False)

import plotly.express as px





fig = px.bar(medals_countries, x="Team", y="Medals", color="Medals",

  animation_frame="Year", animation_group="Team", range_y=[0,500],width=800,height=400,range_x=[0,11])



fig.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),

    paper_bgcolor="white",title="Total Number of Medals Each Country Have with respect to Year"

)

fig.show()

#We can see total number of medals that countries won in each year from 2016 to 1896
from wordcloud import WordCloud #To creat a word cloud
#Preparing Data

athletes=data.iloc[:,1]#getting column that include names and surnames of athletes

list_athletes=list(athletes.unique())

list_athletes=[x.split() for x in list_athletes]

text=""

for i in list_athletes:

    text+= " ".join(i)+" "

        
#Creation of WordCloud

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                min_font_size = 10,max_words=50).generate(text) 



plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 

plt.savefig("picture3.png")



#We can see that names like John, Robert, William are common among athletes
data.head()
#Prepare data

cities=list(data["City"].unique())

number=[]

for c in cities:

    num=len(data[data["City"]==c])

    number.append(num)

olympics=pd.DataFrame(zip(cities,number),columns=["Cities","Total_Number_of_Olympics"])

#We need locations of cities to plot in map

locations=[[41.385063,2.173404],[51.507351,-0.121350],[51.219448,4.402464],[48.856613,2.352222],[51.040460,-114.064167],[45.670448,6.396301],[61.115244,10.466263],[34.075611,-118.297007],[40.754621,-111.902293],[60.169846,24.938383],[44.277978,-73.983374],[-33.877757,151.210423],[33.761983,-84.394615],[59.329344,18.068761],[43.586784,39.721302],[36.639257,138.139658],[45.069010,7.680472],[39.912139,116.410544],[-22.933838,-43.203143],[37.986588,23.727988],[39.197519,-120.235418],[47.287108,11.365400],[43.871561,18.415122],[19.404394,-99.160998],[48.155810,11.538132],[37.551010,126.986137],[52.513924,13.374411],[59.913654,10.752023],[46.536867,12.138926],[-37.815708,144.962830],[41.889501,12.495944],[52.368971,4.870650],[45.536212,-73.629094],[55.745989,37.621592],[35.676647,139.656742],[49.244280,-123.115610],[45.182241,5.724209],[42.974103,141.310834],[45.927709,6.890412],[47.602718,7.542398],[46.500265,9.816901],[47.489244,11.093460]]#Start with mexico city   

olympics["lat"]=[locations[i][0] for i in range(len(locations))]

olympics["lon"]=[locations[i][1] for i in range(len(locations))]
#Visualisation by Bubble Plot

fig = px.scatter_geo(olympics, lat="lat",lon="lon", color="Cities",

                     hover_name="Cities", size="Total_Number_of_Olympics",

                     projection="natural earth")

fig.update_layout(

    title={

        'text':"Number of Olympics which are Hosted by Each Cities",

        'y':0.95,

        'x':0.45,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()

plt.savefig("picture1.png")


number=[]

sports=list(data["Sport"].unique())

for s in sports:

    number.append(len(data[data["Sport"]==s]))

df=pd.DataFrame(zip(sports,number),columns=["Sport","Number"])
fig = px.pie(df, values='Number', names='Sport', title='Population of European continent')

fig.show()

plt.savefig("picture2.png")