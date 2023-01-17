import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import pandas as pd
%matplotlib inline


plt.figure(figsize=(15,10))
img = np.array(Image.open(r"../input/mom2018/mom.jpg"))
plt.imshow(img,interpolation="bilinear")
plt.axis("off")
plt.show()
#importing libraries
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud,STOPWORDS
import io
import base64
from matplotlib import rc,animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
import os
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import random
import cufflinks as cf 
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
#print(os.listdir("../input"))
from matplotlib_venn import venn2
import plotly.plotly as py


import random
number_of_colors = 1000
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
color_theme = dict(color = color)

#---------------------Bar Plot----------------------------------------#
def plot_insight(x,y,header="",xaxis="",yaxis=""):
    trace = go.Bar(y=y,x=x,marker=color_theme)
    layout = go.Layout(title = header,xaxis=dict(title=xaxis,tickfont=dict(size=13,)),
                       yaxis=dict(title=yaxis,titlefont=dict(size=16),tickfont=dict(size=14)))
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig,filename='basic-bar1')

#---------------------Donutplot----------------------------------------#
def donut_graph(label1, value1, label2, value2,title = "",text1="",text2=""):
    fig = {
        "data": [{
                  "values": value1,
                  "labels": label1,
                  "domain": {'x': [0.0, 0.35], 'y': [2.22, 2.53]},
                  "name": text1,
                  "hoverinfo":"label+percent+name",
                  "hole": .45,  
                  "type": "pie",
                },     
                {
                  "values": value2,
                  "labels": label2,
                  "text":text2,
                  "textposition":"inside",
                  "domain":{'x': [0.50, 0.85], 'y': [2.22, 2.53]},
                  "name": text2,
                  "hoverinfo":"label+percent+name",
                  "hole": .45,  
                  "type": "pie",
                  "textinfo": value2
                }],
        "layout": {
                    "title":title,
                    "annotations": [
                        {
                            "font": {
                                "size": 12,
                            },
                            "showarrow": False,
                            "text": text1,
                            "x": 0.12,
                            "y": 0.5
                        },
                        {
                            "font": {
                                "size": 12,
                            },
                            "showarrow": False,
                            "text": text2,
                            "x": 0.73,
                            "y": 0.5
                            }]
                }
            }
    return iplot(fig, filename='donut')
#---------------------Word Cloud Plot----------------------------------------#
def word_cloud_graph(df):
    # data prepararion
    plt.subplots(figsize=(20,12))
    wordcloud = WordCloud(background_color='white',width=512,height=384,).generate(" ".join(df))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('graph.png')
    return plt.show()
#-------------------Ven Diagram--------------------------------#
def venn(team1,team2):
    plt.figure(figsize=(20,10))
    venn2([set(fifa[fifa["Team"] == team1]["Opponent"]), set(fifa[fifa["Team"] == team2]["Opponent"])], set_labels = ('Croatia', 'Belgium'))
    return plt.show()

# fifa[fifa["Team"] == "Croatia"]["Opponent"],fifa[fifa["Team"] == "Belgium"]["Opponent"]
fifa = pd.read_csv("../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv")
fifa.head()
goals = fifa.groupby('Team')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'),ascending=False)
plot_insight(x=goals["Team"],y=goals["Goal Scored"],header="Total Goal Done by team", xaxis="Teams",yaxis="Total Goals")
label1=goals["Team"]
value1=(goals["Goal Scored"]/goals["Goal Scored"].sum())*100
label2 = goals["Team"]
value2=(goals["Goal Scored"]/goals["Goal Scored"].sum())
donut_graph(label1, value1, label2, value2,title ="Total Goals By Teams",text1="Goal Percent",text2="Goal Count")
word_cloud_graph(goals["Team"])
attempt = fifa.groupby('Team')['Attempts'].sum().reset_index().sort_values(by=('Attempts'),ascending=False)
plot_insight(x=attempt["Team"],y=attempt["Attempts"],header="Total Goal Attempts Done by team", xaxis="Teams",yaxis="Total attempt")
label1=attempt["Team"]
value1=(attempt["Attempts"]/attempt["Attempts"].sum())*100
label2 = attempt["Team"]
value2= attempt["Attempts"].value_counts()
donut_graph(label1, value1, label2, value2,title ="Total Goals Attempt By Teams",text1="Attempts Percent",text2="Attempt Count")
word_cloud_graph(attempt["Team"])
ball_possession=fifa.groupby('Team')['Ball Possession %'].sum().reset_index().sort_values(by=('Ball Possession %'),ascending=False)
plot_insight(x=ball_possession["Team"],y=ball_possession["Ball Possession %"],header="Total Ball Possession % Done by team", xaxis="Teams",yaxis="Total ball_possession")
label1=ball_possession["Team"]
value1=(ball_possession["Ball Possession %"]/ball_possession["Ball Possession %"].sum())*100
label2 = ball_possession["Team"]
value2=(ball_possession["Ball Possession %"].values)
donut_graph(label1, value1, label2, value2,title ="Total Goals Attempt By Teams",text1="Ball Positions Percent",text2="Ball Positions Count")
word_cloud_graph(ball_possession["Team"])
# Plotting total Man of the Match awards for teams

# Encoding the values for the column man of the Match
mom_1={'Man of the Match':{'Yes':1,'No':0}}
fifa.replace(mom_1,inplace=True)

# Converting column datatype to int
fifa['Man of the Match']=fifa['Man of the Match'].astype(int)

mom=fifa.groupby('Team')['Man of the Match'].sum().reset_index().sort_values(by=('Man of the Match'),ascending=False)
plot_insight(x=mom["Team"],y=mom["Man of the Match"],header="Total Man of the Match by team", xaxis="Teams",yaxis="Total Man of the Match")
label1=mom["Team"]
value1=(mom["Man of the Match"]/mom["Man of the Match"].sum())*100
label2 = mom["Team"]
value2=(mom["Man of the Match"]/mom["Man of the Match"].sum())
donut_graph(label1, value1, label2, value2,title ="Total Man of The Match By Teams",text1="MOM Percent",text2="MOM Count")
Blocked = fifa.groupby('Team')['Blocked'].sum().reset_index().sort_values(by=('Blocked'),ascending=False)
plot_insight(x=Blocked["Team"],y=Blocked["Blocked"],header="Total Blocked by team", xaxis="Teams",yaxis="Total Blocked")
word_cloud_graph(Blocked["Team"])
for i in fifa["Team"].unique():
    df = fifa[fifa["Team"] == i][["Opponent","Goal Scored"]]
    plot_insight(x=df["Opponent"],y=df["Goal Scored"],header= i +" Goals Against Opponent", xaxis="Teams",yaxis="Goals")
group_attempt = fifa.groupby('Team')['On-Target','Off-Target','Blocked'].sum().reset_index()
group_attempt_sorted = group_attempt.melt('Team', var_name='Target', value_name='Value')
# Plotting the new dataset created above
plt.figure(figsize = (20, 8), facecolor = "White")

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=group_attempt_sorted, color = color[100])

plot1.set_xticklabels(group_attempt_sorted['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Attempts')
plot1.set_title('Total On-Target, Off-Target and Blocked attempts by teams')
for i in group_attempt_sorted["Team"].unique():
    df = group_attempt_sorted[group_attempt_sorted["Team"] == i]
    plot_insight(x=df["Target"],y=df["Value"],header= i +" On-Target, Off-Target, Bloaked Attempts", xaxis="Teams",yaxis="Goals")
saves=fifa.groupby('Team')['Saves'].sum().reset_index().sort_values(by=('Saves'),ascending=False)
plot_insight(x=saves["Team"],y=saves["Saves"],header="Total Saves Goals by team", xaxis="Teams",yaxis="Save goal")
label1=saves["Team"]
value1=(saves["Saves"]/saves["Saves"].sum())*100
label2 = saves["Team"]
value2=(saves["Saves"].values)
donut_graph(label1, value1, label2, value2,title ="Total Goals Saved By Teams",text1="Save Percent",text2="Save Count")
word_cloud_graph(saves["Team"])
own_goal=fifa.groupby('Opponent')['Own goals'].sum().reset_index().sort_values(by=('Own goals'),ascending=False)
own_goal=own_goal[own_goal['Own goals']!=0]
plot_insight(x=own_goal["Opponent"],y=own_goal["Own goals"],header="Teams who did Own goals against themselves", xaxis="Opponents",yaxis="own_goal")

label1=own_goal["Opponent"]
value1=(own_goal["Own goals"]/own_goal["Own goals"].sum())*100
label2 = own_goal["Opponent"]
value2=(own_goal["Own goals"].values)
donut_graph(label1, value1, label2, value2,title ="Teams who did Own goals against themselves",text1="Save Percent",text2="Save Count")

word_cloud_graph(own_goal["Opponent"])
# Plot of total corners, free kicks and offsides for teams

corners_offsides_freekicks = fifa.groupby('Team')['Corners','Offsides','Free Kicks'].sum().reset_index()
corners_offsides_freekicks

# Changing the dataframe for plotting
corners_offsides_freekicks_sort = corners_offsides_freekicks.melt('Team', var_name='Target', value_name='Value')

# Plotting the new dataset created above
plt.figure(figsize = (20, 8), facecolor = None)

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=corners_offsides_freekicks_sort, color = color[905])

plot1.set_xticklabels(corners_offsides_freekicks_sort['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Totals')
plot1.set_title('Total Corners, free kicks and offsides for teams')
passes=fifa.groupby('Team')['Passes'].sum().reset_index().sort_values(by=('Passes'),ascending=False)
plot_insight(x=passes["Team"],y=passes["Passes"],header="Total passes By Team", xaxis="Teams",yaxis="passes")

label1=passes["Team"]
value1=(passes["Passes"]/passes["Passes"].sum())*100
label2 = passes["Team"]
value2=(passes["Passes"].values)
donut_graph(label1, value1, label2, value2,title ="Total passes By Team",text1="Passes Percent",text2="Passes Count")

word_cloud_graph(passes["Team"])
yellow_cards = fifa.groupby('Team')['Yellow Card'].sum().reset_index().sort_values(by=('Yellow Card'), ascending=False)
red_cards = fifa.groupby('Team')['Red'].sum().reset_index().sort_values(by=('Red'), ascending=False)
plot_insight(x=yellow_cards["Team"],y=yellow_cards["Yellow Card"],header="Total Yellow Card By Team", xaxis="Teams",yaxis="Yellow Card")
plot_insight(x=red_cards["Team"],y=red_cards["Red"],header="Total Red Card By Team", xaxis="Teams",yaxis="Red Card")

label1=yellow_cards["Team"]
value1=(yellow_cards["Yellow Card"]/yellow_cards["Yellow Card"].sum())*100
label2 = yellow_cards["Team"]
value2=(yellow_cards["Yellow Card"].values)
donut_graph(label1, value1, label2, value2,title ="Total Yellow Card By Team",text1="Yellow Card Percent",text2="Yellow Card Count")

label1=red_cards["Team"]
value1=(red_cards["Red"]/red_cards["Red"].sum())*100
label2 = red_cards["Team"]
value2=(red_cards["Red"].values)
donut_graph(label1, value1, label2, value2,title ="Total Red Card By Team",text1="Red Card Percent",text2="Red Card Count")

word_cloud_graph(yellow_cards["Team"])
word_cloud_graph(red_cards["Team"])
# Lables for the Radar plot

labels=np.array(['Goal Scored', 'Attempts', 'Corners', 'Offsides', 'Free Kicks', 'Saves', 'Fouls Committed', 'Yellow Card'])

# Radar data for the Finals, "France vs Croatia"

data1=fifa.loc[126,labels].values
data2=fifa.loc[127,labels].values

# Radar data for Semi-Final 1 - "France vs Belgium"
data3=fifa.loc[120,labels].values
data4=fifa.loc[121,labels].values

# Radar data for Semi-Final 2 - "Croatia vs England"

data5=fifa.loc[122,labels].values
data6=fifa.loc[123,labels].values

# Radar data for 3nd runnerups

data7 = fifa.loc[125,labels].values
data8 = fifa.loc[124,labels].values
# Create a radar plot for Semi-Final 2 using plotly
import plotly
plotly.offline.init_notebook_mode(connected=True)

def radar_chart(data1,data2,name = "", name1 = ""):
    data = [go.Scatterpolar(r = data1,theta = labels,mode = 'lines',fill = 'toself',name = name),
        go.Scatterpolar(r = data2,theta = labels,mode = 'lines',fill = 'toself',name = name1)]
    layout = go.Layout(title = name +" vs "+name1,polar = dict(radialaxis = dict(visible = True,range = [0, 20])),showlegend = True)
    fig = go.Figure(data=data, layout=layout)
    return plotly.offline.iplot(fig)

radar_chart(data3,data4,name = "Final Match: France", name1 ="Belgium")
radar_chart(data1,data2,name="Semifinal Match: France", name1="Croatia")
radar_chart(data5,data6,name="Semifinal Match: Croatia", name1="England")
radar_chart(data7,data8,name="1st Runner up Match: England", name1="Belgium")
