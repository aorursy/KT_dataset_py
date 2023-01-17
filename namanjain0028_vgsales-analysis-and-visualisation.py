import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import plotly.express as px 

import plotly.graph_objs as go 

# plt.style.use("fivethirtyeight")


df = pd.read_csv("../input/videogamesales/vgsales.csv")

df.head(10)
df.describe()
df.info()
print(df.isna().sum())
# as we have a sufficient amount of data we can remove those rows which have missing data as it won't affect overall data much 



print(df.shape)



# deleting the rows having missing values 

df = df.dropna(subset = ["Year" , "Publisher"])



print(df.shape)


grouped = df.groupby(df.Year)[["Global_Sales"]].sum()

grouped = grouped.sort_values(by = "Global_Sales" , ascending = False)

grouped = grouped.head(10)



# plottng 



fig = px.pie(data_frame = grouped , 

            names = grouped.index , 

            values = "Global_Sales" , 

            template = "seaborn" , 

            hole = 0.4 , 

            color_discrete_sequence = px.colors.sequential.Inferno , 

            )



fig.update_layout(title = "Top 10 years for gaming market", 

                  paper_bgcolor = "rgb(230,230,230)" , 

                 plot_bgcolor = "rgb(243,243,243)" , 

                 annotations= [dict(text = "Global Sales" , font_size = 20 , showarrow = False , opacity = 0.7)])



fig.update_traces (rotation = 90 , pull = 0.03 , textinfo = "percent+label")

fig.show()
df1 = pd.crosstab(df.Platform , df.Genre , margins_name = "Platform_Total" , margins = True)

df1 = df1.sort_values(by = "Platform_Total" , ascending = False)

df1 = df1.head(11)

df1 = df1.drop(index = "Platform_Total")



plt.figure(figsize= (15 , 7))



with sns.color_palette(palette = "plasma", n_colors=10, desat=.8):

    fig = sns.barplot(data = df1 , x = df1.index , y = "Platform_Total" , saturation = 0.5)

    

for p in fig.patches : 

    fig.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2, p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points' , )





plt.title("top 10 platform preffered by the publishers")


df2 = pd.crosstab(df.Genre , df.Platform , margins_name = "Platform_Total" , margins = True)

df2 = df2.sort_values(by = "Platform_Total" , ascending = False)

df2 = df2.drop(index = "Platform_Total" , axis = 0) 



fig = px.pie(data_frame = df2 , 

            names = df2.index , 

            values = "Platform_Total" , 

            template = "seaborn" , 

            hole = 0.3 , 

            color_discrete_sequence = px.colors.sequential.Viridis , 

            )



fig.update_layout(title = "Percentage of games belonging each type of Genre", 

                  paper_bgcolor = "rgb(230,230,230)" , 

                 plot_bgcolor = "rgb(243,243,243)" , 

                 annotations= [dict(text = "Genre" , font_size = 20 , showarrow = False , opacity = 0.7)])



fig.update_traces (rotation = 90 , pull = 0.03 , textinfo = "percent+label")

fig.show()
 

df3 = pd.crosstab(df.Publisher , df.Year , margins = True , margins_name = "publisher_total" )

df3 = df3.sort_values(by = "publisher_total" , ascending = False)

df3 = df3.drop(index = "publisher_total" , axis = 0) 

df3 = df3.head(5)



plt.figure(figsize= (15 , 7))

plt.style.use("fivethirtyeight")

plt.title("Top 5 companies that have produced highest number of games over the years")

plt.ylabel("Number of Games")

with sns.color_palette(palette = "Spectral_r", n_colors=10, desat=.8):

    fig = sns.barplot(data = df3 , x = df3.index , y = "publisher_total" , saturation = 0.5)



for p in fig.patches : 

    fig.annotate(format(p.get_height(), '.1f'), 

                   (p.get_x() + p.get_width() / 2, p.get_height()), 

                   ha = 'center', va = 'center', 

                   xytext = (0, 9), 

                   textcoords = 'offset points' , )


df_ea = df[df["Publisher"] == "Electronic Arts"].groupby("Year")[["Global_Sales"]].sum()

df_act = df[df["Publisher"] == "Activision"].groupby("Year")[["Global_Sales"]].sum()

df_namco = df[df["Publisher"] == "Namco Bandai Games"].groupby("Year")[["Global_Sales"]].sum()

df_ubisoft = df[df["Publisher"] == "Ubisoft"].groupby("Year")[["Global_Sales"]].sum()

df_konami = df[df["Publisher"] == "Konami Digital Entertainment"].groupby("Year")[["Global_Sales"]].sum()



fig = go.Figure()



fig.add_trace(go.Scatter(x=df_ea.index, y=df_ea["Global_Sales"],

                    mode='lines+markers',

                    name='Electronic Arts' , 

                    opacity = 0.8 , 

                    fill = "tozeroy" , 

                    marker = dict(color = px.colors.sequential.ice[7])))



fig.add_trace(go.Scatter(x=df_act.index, y=df_act["Global_Sales"],

                    mode='lines+markers',

                    name='Activision' , 

                    opacity = 0.8 , 

                    visible = "legendonly" , 

                        fill = "tozeroy"))



fig.add_trace(go.Scatter(x=df_namco.index, y=df_namco["Global_Sales"],

                    mode='lines+markers',

                    name='Namco', 

                    opacity = 0.8 , 

                    visible = "legendonly" , 

                        fill = "tozeroy"))



fig.add_trace(go.Scatter(x=df_ubisoft.index, y=df_ubisoft["Global_Sales"],

                    mode='lines+markers',

                    name='Ubisoft' , 

                    opacity = 0.8 , 

                    visible = "legendonly" , 

                        fill = "tozeroy"))



fig.add_trace(go.Scatter(x=df_konami.index, y=df_konami["Global_Sales"],

                    mode='lines+markers',

                    name='Konami Digital Entertainment' , 

                    opacity = 0.8, 

                    visible = "legendonly" , 

                        fill = "tozeroy"))





fig.update_layout(title = "Yearly Global Sales of above 5 Publishers" , 

                 xaxis = dict(title = "Year" , 

                         gridcolor = "white" ,

                         gridwidth = 2) , 

                 yaxis = dict(title = "Global Sales" , 

                         gridcolor = "white" , 

                         gridwidth = 2) ,

                 paper_bgcolor = "rgb(225,225,225)" , 

                 plot_bgcolor  = "rgb(243,243,243)" , 

                 legend = dict(orientation = "h",

                               yanchor="bottom",

                               y=1.02,

                               xanchor="right",

                               x=1)  , 

                 height = 600)



fig.show()



df_4 = pd.crosstab(df.Year , df.Platform , margins = True , margins_name= "games_developed").drop(index = "games_developed", axis = 0)[["games_developed"]]



fig = go.Figure()



fig.add_trace(go.Scatter(x = df_4.index , 

                        y = df_4["games_developed"] , 

                        name = "Games" , 

                        opacity = 0.8 , 

                        mode = "markers+lines" ,

                        marker = dict(color = "violet") , 

                        visible = True , 

                        fillcolor = "rgba(150,150,150)" , 

                        fill = "tonexty"))





fig.update_layout(title = "Game production over the years" , 

                 xaxis = dict(title = "Years" , 

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 yaxis = dict(title = "Number of games" , 

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 paper_bgcolor = 'rgb(225,225,225)' , 

                 plot_bgcolor = "rgb(243,243,243)" , 

                 showlegend = True , 

                 legend = dict(orientation = "h",

                               yanchor="bottom",

                               y=1.02,

                               xanchor="right",

                               x=1))

fig.show()
fig = px.scatter_3d(data_frame = df[df["Year"] > 2000].sort_values(by = "Year" , ascending = True) , 

                  x = "NA_Sales" , 

                  y = "EU_Sales" , 

                  z = "JP_Sales" , 

                  labels = {

                      "NA_Sales" : "NA Sales" , 

                      "EU_Sales" : "EU Sales" , 

                      "JP_Sales" : "JP Sales"

                  } , 

                  animation_frame = "Year" , 

                  size = "Other_Sales" , 

                  width = 800 , 

                  height = 600 , 

                  size_max = 50 , 

                  opacity = 0.8 , 

                  color = "Genre")





fig.update_layout(title = "3D scatter plot between North America , Europe and Japan Sales" , 

                  

                 paper_bgcolor = 'rgb(230,230,230)' , 

                 plot_bgcolor = "rgb(243,243,243)" , 

                 showlegend = True)

fig.show()


df_gs = df.groupby("Publisher")[["Global_Sales"]].sum().sort_values(by = "Global_Sales" , ascending = False).head(10)



fig = px.pie(data_frame = df_gs , 

            names = df_gs.index , 

            values = "Global_Sales" , 

            template = "seaborn" , 

            hole = 0.3 , 

            color_discrete_sequence = px.colors.sequential.Inferno , 

            )



fig.update_layout(title = "Top 10 companies with highest global sales over the years ", 

                  paper_bgcolor = "rgb(230,230,230)" , 

                 plot_bgcolor = "rgb(243,243,243)" , 

                 annotations= [dict(text = "Top 10 " , font_size = 20 , showarrow = False , opacity = 0.7)])



fig.update_traces (rotation = 90 , pull = 0.03 , textinfo = "percent+label")

fig.show()
df_ea = df[df["Publisher"] == "Electronic Arts"].groupby("Year")[["Global_Sales"]].sum()

df_act = df[df["Publisher"] == "Activision"].groupby("Year")[["Global_Sales"]].sum()

df_namco = df[df["Publisher"] == "Namco Bandai Games"].groupby("Year")[["Global_Sales"]].sum()

df_ubisoft = df[df["Publisher"] == "Ubisoft"].groupby("Year")[["Global_Sales"]].sum()

df_konami = df[df["Publisher"] == "Konami Digital Entertainment"].groupby("Year")[["Global_Sales"]].sum()



df_nintendo = df[df["Publisher"] == "Nintendo"].groupby("Year")[["Global_Sales"]].sum()

df_tti = df[df["Publisher"] == "Take-Two Interactive"].groupby("Year")[["Global_Sales"]].sum()

df_sony = df[df["Publisher"] == "Sony Computer Entertainment"].groupby("Year")[["Global_Sales"]].sum()

df_sega = df[df["Publisher"] == "Sega"].groupby("Year")[["Global_Sales"]].sum()

df_thq= df[df["Publisher"] == "TQ"].groupby("Year")[["Global_Sales"]].sum()





fig = go.Figure()



fig.add_trace(go.Scatter(x=df_ea.index, y=df_ea["Global_Sales"],

                    mode='lines+markers',

                    name='Electronic Arts' , 

                    opacity = 0.8 , 

                    ))



fig.add_trace(go.Scatter(x=df_act.index, y=df_act["Global_Sales"],

                    mode='lines+markers',

                    name='Activision' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_namco.index, y=df_namco["Global_Sales"],

                    mode='lines+markers',

                    name='Namco', 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_ubisoft.index, y=df_ubisoft["Global_Sales"],

                    mode='lines+markers',

                    name='Ubisoft' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_konami.index, y=df_konami["Global_Sales"],

                    mode='lines+markers',

                    name='Konami Digital Entertainment' , 

                    opacity = 0.8, 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_nintendo.index, y=df_nintendo["Global_Sales"],

                    mode='lines+markers',

                    name='Nintendo' , 

                    opacity = 0.8 , 

                    visible = "legendonly")) 



fig.add_trace(go.Scatter(x=df_tti.index, y=df_tti["Global_Sales"],

                    mode='lines+markers',

                    name='Take-Two Interactive' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_sony.index, y=df_sony["Global_Sales"],

                    mode='lines+markers',

                    name='Sony Computer Entertainment' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_sega.index, y=df_sega["Global_Sales"],

                    mode='lines+markers',

                    name='Sega' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))



fig.add_trace(go.Scatter(x=df_thq.index, y=df_thq["Global_Sales"],

                    mode='lines+markers',

                    name='THQ' , 

                    opacity = 0.8 , 

                    visible = "legendonly"))





fig.update_layout(title = "Progress chart of top 10 companies having highest total global sales" , 

                 xaxis = dict(title = "Year" , 

                         gridcolor = "white" , 

                         gridwidth = 2) , 

                 yaxis = dict(title = "Global Sales" , 

                         gridcolor = "white" , 

                         gridwidth = 2) ,

                 paper_bgcolor = "rgb(225,225,225)" , 

                 plot_bgcolor  = "rgb(243,243,243)" , 

               

                 height = 600)



fig.show()
