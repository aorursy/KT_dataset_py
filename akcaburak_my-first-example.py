# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from collections import Counter

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_game=pd.read_csv("../input/vgsales.csv")
data_game.info()
year_list = [2006., 1985., 2008., 2009., 1996., 1989., 1984., 2005., 1999.,

        2007., 2010., 2013., 2004., 1990., 1988., 2002., 2001., 2011.,

        1998., 2015., 2012., 2014., 1992., 1997., 1993., 1994., 1982.,

        2003., 1986., 2000., 1995., 2016., 1991., 1981., 1987.,

        1980., 1983., 2020., 2017.]

global_sales_ratio=[]



for each in year_list:

    x=data_game[data_game["Year"] == each]

    sales_rate=sum(x.Global_Sales)/len(x)

    global_sales_ratio.append(sales_rate)

data = pd.DataFrame ( { "Year" : year_list, "Ratio_of_Sales" : global_sales_ratio})

new_index=(data["Ratio_of_Sales"].sort_values(ascending=False).index.values)

sorted_data=data.reindex(new_index)





plt.figure(figsize=(15,8))

sns.barplot(x=sorted_data["Year"] , y=sorted_data["Ratio_of_Sales"])

plt.xticks(rotation=90)

plt.title("Mean of Games Sales Revenue By Year",size=30,style="italic")

plt.xlabel("Years",size=20)

plt.ylabel("Mean of Sales Revenue",size=20)

plt.show()





game_genre_list=list(data_game.Genre.unique())

mean_of_genre=[]



for each in game_genre_list:

    x=data_game[data_game.Genre==each]

    rate_genre=sum(x.Global_Sales)/len(x)

    mean_of_genre.append(rate_genre)



data=pd.DataFrame({"Genre":game_genre_list  , "mean_of_genre" : mean_of_genre   })

new_index=data.mean_of_genre.sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)





plt.figure(figsize=(15,8))

sns.barplot(x="Genre",y="mean_of_genre",data=sorted_data)

plt.xticks(rotation=45)

plt.title("Mean of Games Global Sales Revenue by Games Genre",size=30)

plt.xlabel("Games Genre",size=20)

plt.ylabel("Mean of Revenue",size=20)

plt.show()

game_genre_list=list(data_game.Genre.unique())

mean_of_genre=[]



for each in game_genre_list:

    x=data_game[data_game.Genre==each]

    rate_genre=sum(x.Global_Sales)/len(x)

    mean_of_genre.append(rate_genre)



data=pd.DataFrame({"Genre":game_genre_list  , "mean_of_genre" : mean_of_genre   })

new_index=data.mean_of_genre.sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)





labels = sorted_data.Genre

values=sorted_data.mean_of_genre



trace1=go.Pie(

        labels=labels,

        values=values,

        hoverinfo="label+percent+name",

        textfont=dict(size=15),

        text=sorted_data.Genre,

        hole=0.02,

        marker=dict( line = dict(color='#000000', width=2)))



data=[trace1]





layout=dict(title="distribution of game types")



fig=dict(data=data,layout=layout)

iplot(fig)
platform_list=list(data_game.Platform.unique())

ratio_platform=[]



for each in platform_list:

    x=data_game[data_game.Platform==each]

    rate_of_platform=sum(x.Global_Sales)/len(x)

    ratio_platform.append(rate_of_platform)



data=pd.DataFrame({"Platform":platform_list  ,  "Sales_platform":ratio_platform})

new_index=data.Sales_platform.sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)





plt.figure(figsize=(20,8))

sns.barplot(x="Platform",y="Sales_platform",data=sorted_data)

plt.xticks(rotation=45)

plt.title("Mean of Sales amount according to Platform",size=30,style="italic")

plt.xlabel("Platform",size=20)

plt.ylabel("Mean of Sales " , size=20)

plt.show()

pc=data_game[data_game.Platform=="PC"]

wii=data_game[data_game.Platform=="Wii"]

ps=data_game[data_game.Platform=="PS4"]

pc_wii_ps4=pd.concat([pc,wii,ps],axis=0,ignore_index=True)

data_pc_wii_ps4=data_game[(data_game.Platform=="PC")|(data_game.Platform=="Wii")|(data_game.Platform=="PS4")]

data_pc_wii_ps4=data_pc_wii_ps4[(data_pc_wii_ps4.Genre=="Action")|(data_pc_wii_ps4.Genre=="Sports")|(data_pc_wii_ps4.Genre=="Strategy")|(data_pc_wii_ps4.Genre=="Racing")]
genre_list=list(data_pc_wii_ps4.Genre.unique())

platform_list=list(data_pc_wii_ps4.Platform.unique())



platform_sports=data_pc_wii_ps4[data_pc_wii_ps4.Genre=="Sports"].Platform.value_counts()

platform_racing=data_pc_wii_ps4[data_pc_wii_ps4.Genre=="Racing"].Platform.value_counts()

platform_action=data_pc_wii_ps4[data_pc_wii_ps4.Genre=="Action"].Platform.value_counts()

platform_strategy=data_pc_wii_ps4[data_pc_wii_ps4.Genre=="Strategy"].Platform.value_counts()
data_new_plat_genre=pd.DataFrame({"Sports":platform_sports  ,  "Racing" : platform_racing , "Action":platform_action , "Strategy" :platform_strategy })

df_pc=data_new_plat_genre.loc["PC",:]

df_wii=data_new_plat_genre.loc["Wii",:]

df_ps4=data_new_plat_genre.loc["PS4",:]

trace1=go.Bar(

        x=data_new_plat_genre.columns,

        y=df_pc,

        name="Pc",

        marker=dict(color="rgba(255,55,0,0.7)",line=dict(color="rgb(0,0,0)",width=1.5))

        

)

trace2=go.Bar(

        x=data_new_plat_genre.columns,

        y=df_wii,

        name="Wii",

        marker=dict(color="rgba(100,155,110,0.7)",line=dict(color="rgb(0,0,0)",width=1.5))

        

)



trace3=go.Bar(

        x=data_new_plat_genre.columns,

        y=df_ps4,

        name="PS4",

        marker=dict(color="rgba(200,100,225,0.7)",line=dict(color="rgb(0,0,0)",width=1.5))

        

)



layout=go.Layout(

            title="Sales Count of Most Popular Games Genre By Platform ",

            barmode="group",

            xaxis=dict(title="Genre",zeroline=False, tickfont=dict(size=40,color='rgb(107, 107, 107)')),

            

            

            yaxis=dict(title=" Count ",zeroline=False)

  )







data=[trace1,trace2,trace3]

fig=go.Figure(data=data,layout=layout)

iplot(fig)

data_game.head(1)
data_pc=data_game[data_game.Platform=="PC"]

data_wii=data_game[data_game.Platform=="Wii"]

data_ps4=data_game[data_game.Platform=="PS4"]



count_pc_jp=sum(data_pc.JP_Sales)/len(data_pc)

count_wii_jp=sum(data_wii.JP_Sales)/len(data_wii)

count_ps4_jp=sum(data_ps4.JP_Sales)/len(data_ps4)





count_pc_eu=sum(data_pc.EU_Sales)/len(data_pc)

count_wii_eu=sum(data_wii.EU_Sales)/len(data_wii)

count_ps4_eu=sum(data_ps4.EU_Sales)/len(data_ps4)





count_pc_oth=sum(data_pc.Other_Sales)/len(data_pc)

count_wii_oth=sum(data_wii.Other_Sales)/len(data_wii)

count_ps4_oth=sum(data_ps4.Other_Sales)/len(data_ps4)







data_platform=["Pc","Wii","PS4"]

data_count_jp=[count_pc_jp ,count_wii_jp , count_ps4_jp]

data_count_eu=[count_pc_eu , count_wii_eu , count_ps4_eu]

data_count_oth=[count_pc_oth , count_wii_oth , count_ps4_oth]



trace1=go.Bar(

        x=data_platform,

        y=data_count_jp,

        name="JP",

        marker=dict(color="rgba(255,87,0,.7)",line=dict(color="rgb(0,0,0)",width=1))

        )



trace2=go.Bar(

        x=data_platform,

        y=data_count_eu,

        name="EU",

        marker=dict(color="rgba(200,187,255,.7)",line=dict(color="rgb(0,0,0)",width=1))

)



trace3=go.Bar(

        x=data_platform,

        y=data_count_oth,

        name="Others",

        marker=dict(color="rgba(100,17,150,.7)",line=dict(color="rgb(0,0,0)",width=1))

)

data=[trace1,trace2,trace3]





layout=go.Layout(

            title="Distribution Of Sales by Region",

            barmode="group",

            xaxis=dict(title="Platform",zeroline=False, tickfont=dict(size=40,color='rgb(107, 107, 107)')),

            

            

            yaxis=dict(title="Ratio of Games Sales ",zeroline=False)

  )



fig=go.Figure(data=data,layout=layout)





iplot(fig)



data_wii=data_game[data_game.Platform=="Wii"]

data_pc=data_game[data_game.Platform=="PC"]

data_ps=data_game[(data_game.Platform=="PS")|(data_game.Platform=="PS3")|(data_game.Platform=="PS4")]



new_index_pc=(data_pc.Global_Sales.sort_values(ascending=False)).index.values

sorted_data_pc=data_pc.reindex(new_index_pc)





new_index_wii=(data_wii.Global_Sales.sort_values(ascending=False)).index.values

sorted_data_wii=data_wii.reindex(new_index_wii)



new_index_ps=(data_ps.Global_Sales.sort_values(ascending=False)).index.values

sorted_data_ps=data_ps.reindex(new_index_ps)



datapc=sorted_data_pc.iloc[0:3,:]

datawii=sorted_data_wii.iloc[0:3,:]

dataps=sorted_data_ps.iloc[0:3,:]







trace1=go.Bar(

            x=["Pc","Wii","Ps"],

            y=[datapc.Global_Sales.iloc[0],datawii.Global_Sales.iloc[0],dataps.Global_Sales.iloc[0]],

            text=[datapc.Name.iloc[0],datawii.Name.iloc[0],dataps.Name.iloc[0]],

            name="First Games Of Global Sales",

            marker=dict(color="rgba(255,75,0,.7)",line=dict(color="rgb(0,0,0)",width=2))

            

                  )



trace2=go.Bar(

        x=["Pc","Wii","Ps"],

        y=[datapc.Global_Sales.iloc[1],datawii.Global_Sales.iloc[1],dataps.Global_Sales.iloc[1]],

            text=[datapc.Name.iloc[1],datawii.Name.iloc[1],dataps.Name.iloc[1]],

            name="Second Games Of Global Sales",

            marker=dict(color="rgba(100,75,200,.7)",line=dict(color="rgb(0,0,0)",width=2))

)





trace3=go.Bar(

        x=["Pc","Wii","Ps"],

        y=[datapc.Global_Sales.iloc[2],datawii.Global_Sales.iloc[2],dataps.Global_Sales.iloc[2]],

            text=[datapc.Name.iloc[2],datawii.Name.iloc[2],dataps.Name.iloc[2]],

            name="Third Games Of Global Sales",

            marker=dict(color="rgba(0,175,255,.7)",line=dict(color="rgb(0,0,0)",width=2))

)



layout=go.Layout(

        title="First Three Games of Global Sales",

        xaxis=dict(title="Platform",ticklen=5,zeroline=False , tickfont=dict(size=40,color='rgb(107, 107, 107)')),

        yaxis=dict(title="Global Sales",ticklen=5,zeroline=False)

)





data=[trace1,trace2,trace3]

fig=go.Figure(data=data,layout=layout)



iplot(fig)

data_nan=data_game.dropna(axis=0)

publisher_name=list(data_nan.Publisher.unique())

publisher_ratio=[]



for each in publisher_name:

    x=data_nan[data_nan.Publisher==each]

    publisher_rate=sum(x.Global_Sales)/len(x)

    publisher_ratio.append(publisher_rate)



df_publisher=pd.DataFrame({"Publisher":publisher_name , "Publishers_Sales":publisher_ratio})

new_index=df_publisher.Publishers_Sales.sort_values(ascending=False).index.values

sorted_data=df_publisher.reindex(new_index)

df=sorted_data.iloc[0:100,:]



plt.figure(figsize=(40,10))

sns.set_style("darkgrid")



sns.barplot(y=df.Publishers_Sales,x=df.Publisher)

plt.xticks(rotation=90)

plt.xlabel("Publisher",size=20)

plt.ylabel("Sales Ratio",size=20)

plt.title("Mean of Global Sales According to Publisher",size=30)

plt.show()



publisher_count=Counter(data_nan.Publisher)

most_common_publisher=publisher_count.most_common(15)



x,y=zip(*most_common_publisher)

x,y=list(x),list(y)



plt.figure(figsize=(20,10))

sns.barplot(x=x,y=y,palette=sns.dark_palette("red",len(x)))

plt.xlabel("Publisher",fontsize=15)

plt.ylabel("Publish rate",fontsize=15)

plt.title("Top 15 Total Publisher Games Released",fontsize=20,style="oblique")

plt.xticks(rotation=70)

plt.show()





genre_count=Counter(data_nan.Genre)

most_common_genre=genre_count.most_common(15)

x,y=zip(*most_common_genre)

x,y=list(x),list(y)



plt.figure(figsize=(20,10))

sns.barplot(y=x,x=y,palette=sns.dark_palette("yellow",len(x)))

plt.xlabel("Genre Count",fontsize=15)

plt.ylabel("Genre",fontsize=15)

plt.title("Top 15 Total Genre Games Released",fontsize=20,style="oblique")

plt.xticks(rotation=90)

plt.show()
na=sum(data_nan.NA_Sales)

jp=sum(data_nan.JP_Sales)

eu=sum(data_nan.EU_Sales)

oth=sum(data_nan.Other_Sales)

sales_list=[na,eu,jp,oth]

sales_loc=["North America","European Union","Japan","Others"]



plt.figure(figsize=(15,8))

sns.barplot(x=sales_list,y=sales_loc,palette=sns.color_palette("BuGn_r"))

plt.xlabel("Revenue",fontsize=15,style="italic")

plt.ylabel("Region",fontsize=15,style="oblique")

plt.title("Total Revenue by Region",fontsize=20,style="oblique")

plt.xticks(rotation=90)

plt.show()
genre_list=Counter(data_nan.Genre)

genre_list_most=genre_list.most_common(5)

x,y=zip(*genre_list_most)

x,y=list(x),list(y)

liste_na=[]

liste_eu=[]

liste_jp=[]

liste_oth=[]





for each in x:

    genre_data=data_nan[data_nan.Genre==each]

    count_genre_na=sum(genre_data.NA_Sales)

    count_genre_eu=sum(genre_data.EU_Sales)

    count_genre_jp=sum(genre_data.JP_Sales)

    count_genre_oth=sum(genre_data.Other_Sales)

    liste_na.append(count_genre_na)

    liste_eu.append(count_genre_eu)

    liste_jp.append(count_genre_jp)

    liste_oth.append(count_genre_oth)



data_new=pd.DataFrame({"Region":x  ,  "NA": liste_na ,  "EU":liste_eu  , "JP" : count_genre_jp , "OTH": liste_oth })



trace1=go.Bar(

            x=data_new.Region,

            y=data_new.NA,

            name="North America",

            marker=dict(color="rgba(255,80,0,.7)",line=dict(color="rgb(0,0,0)",width=2))

)



trace2=go.Bar(

            x=data_new.Region,

            y=data_new.EU,

            name="European Union",

            marker=dict(color="rgba(100,80,255,.7)",line=dict(color="rgb(0,0,0)",width=2))

)



trace4=go.Bar(

            x=data_new.Region,

            y=data_new.JP,

            name="Japan",

            marker=dict(color="rgba(200,180,0,.7)",line=dict(color="rgb(0,0,0)",width=2))

)



trace3=go.Bar(

            x=data_new.Region,

            y=data_new.OTH,

            name="Others",

            marker=dict(color="rgba(0,255,255,.7)",line=dict(color="rgb(0,0,0)",width=2))

)



data=[trace1,trace2,trace3,trace4]



layout=go.Layout(

                title="Distribution of  Games Genre By Region",

        xaxis=dict(title="Top 3 Genre By Most Sales" , ticklen=5 , zeroline=False , tickfont=dict(size=40,color='rgb(0, 0, 0)')),

        yaxis=dict(title="Global Sales",ticklen=5,zeroline=False)



)

fig=go.Figure(data=data,layout=layout)

iplot(fig)
