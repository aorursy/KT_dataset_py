import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
from wordcloud import WordCloud
warnings.filterwarnings("ignore")
plt.style.use('seaborn')
from PIL import Image
img=np.array(Image.open('../input/wcbrazil/Football.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mat=pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')
mat.head()
ply=pd.read_csv('../input/fifa-world-cup/WorldCupPlayers.csv')
ply.head()
wc=pd.read_csv('../input/clean-data/WorldCups_2.csv')
wc.head()
mat=mat.replace('Germany FR','Germany')
ply=ply.replace('Germany FR','Germany')
wc=wc.replace('Germany FR','Germany')
wc=wc.replace('Croatia ','Croatia')
print('Football world cups were held in years -->',wc['Year'].unique())
#wc['Year'].unique()
print('Number of times World cup football has taken place -->',wc.shape[0])
print('Countries that have hosted the Football world cups -->',wc['Country'].unique())
#wc['Country'].value_counts().plot.pie(shadow=True,startangle=0,explode=(0,0,0,0.2,0,0,0,0,0,0,0,0,0,0,0,0))
plt.title('Which Countries have Hosted world Cup')
#matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, hold=None, data=None)[source]¶
ax=wc['Country'].value_counts().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
plt.xlabel('Country',fontsize=30)
plt.ylabel('No of Time Hosted',fontsize=20)
plt.title('Countried that Hosted World Cup',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()
gold = wc["Winner"]
silver = wc["Runners-Up"]
bronze = wc["Third"]

gold_count = pd.DataFrame.from_dict(gold.value_counts())
silver_count = pd.DataFrame.from_dict(silver.value_counts())
bronze_count = pd.DataFrame.from_dict(bronze.value_counts())
podium_count = gold_count.join(silver_count, how='outer').join(bronze_count, how='outer')
podium_count = podium_count.fillna(0)
podium_count.columns = ['WINNER', 'SECOND', 'THIRD']
podium_count = podium_count.astype('int64')
podium_count = podium_count.sort_values(by=['WINNER', 'SECOND', 'THIRD'], ascending=False)

podium_count.plot(y=['WINNER', 'SECOND', 'THIRD'], kind="bar", 
                  color =['gold','silver','brown'],figsize=(15, 6), fontsize=14,
                 width=.8, align='center')
#plt.xlabel('Countries')
plt.ylabel('Number of World Cup Wins')
plt.title('Countries that won Football World cup')
plt.ioff()
plt.subplots(figsize=(15,6))
ax=wc['Winner'].value_counts().plot.bar(width=0.9)#color=sns.color_palette('summer',20))
ax.set_title('Winner at Football World Cup')
#ax.set_xlabel('Countries')
ax.set_ylabel('Count')
plt.show()
plt.subplots(figsize=(15,6))
ax=wc['Runners-Up'].value_counts().plot.bar(width=0.9)#color=sns.color_palette('summer',20))
ax.set_yticks(np.arange(1,5,1))
ax.set_title('Runnerup at Football World Cup')
#ax.set_xlabel('Countries')
ax.set_ylabel('Count')
plt.show()
wc['Attendance_m']=wc['Attendance']/1000
plt.figure(figsize = (22,12))
#sns.set_style("dark")
plt.subplot(221)
g1 = sns.barplot(x="Year", y="QualifiedTeams", data=wc, palette="spring")
g1.set_title("TEAMS QUALIFIED PER CUP", fontsize=14)

plt.subplot(222)
g2 = sns.barplot(x="Year", y="MatchesPlayed", data=wc, palette="spring")
g2.set_title("NUMBER OF MATCHES PER CUP", fontsize=14)

plt.subplot(223)
g2 = sns.barplot(x="Year", y="Attendance_m", data=wc, palette="spring")
g2.set_title("ATTENDANCE IN MILLIONS PER CUP", fontsize=14)

plt.subplot(224)
g2 = sns.barplot(x="Year", y="GoalsScored", data=wc, palette="spring")
g2.set_title("NUMBER OF GOALS PER CUP", fontsize=14)

plt.subplots_adjust(wspace =.2,hspace =0.4,top = 1)

plt.show()
plt.figure(figsize=(13,7))
ax = plt.scatter("Year","GoalsScored",data=wc,c=wc["GoalsScored"],cmap="inferno",s=900,alpha=.7,linewidth=2,edgecolor="k")
plt.xticks(wc["Year"].unique())
plt.yticks(np.arange(50,210,20))
plt.xlabel('Year',color='r')
plt.ylabel('Goals Scored',color='r')
plt.title('Goals scored Vs Year',color='b')
plt.ioff()
mat = mat.drop_duplicates(subset="MatchID",keep="first")
mat = mat[mat["Year"].notnull()]
mat["Year"] = mat["Year"].astype(int)
mat['total_goals']=mat['Home Team Goals']+mat['Away Team Goals']
plt.figure(figsize=(13,8))
sns.boxplot(y=mat["total_goals"],
            x=mat["Year"])
plt.grid(True)
plt.title("Total goals scored during game by year",color='b')
plt.show()
plt.figure(figsize=(12,7))
sns.barplot(wc['Year'],wc['MatchesPlayed'],color='b',linewidth=2,label='Matches Played')
sns.barplot(wc['Year'],wc['QualifiedTeams'],color='r',linewidth=2,edgecolor='k'*len(wc),label='Teams Participated')
plt.legend(loc='best',prop={'size':13})
plt.title('Teams/Matched by year',color='b')
plt.grid(True)
plt.ylabel('Matches/teams')
plt.ioff()
import networkx as nx 

def interactions(year,color):
    
    df  =  mat[mat["Year"] == year][["Home Team Name","Away Team Name"]]
    G   = nx.from_pandas_dataframe(df,"Home Team Name","Away Team Name")
    
    plt.figure(figsize=(10,9))
    
    nx.draw_kamada_kawai(G,with_labels = True,node_size  = 2500,node_color = color,node_shape = "h",edgecolor  = "k",linewidths  = 5 ,font_size  = 13 ,alpha=.8)
    
    plt.title("Interaction between teams :" + str(year) , fontsize =13 , color = "b")
interactions(2002,"c")
wrds1 = ply["Coach Name"].str.split("(").str[0].value_counts().keys()

wc1 = WordCloud(scale=5,max_words=1000,colormap="rainbow",background_color="black").generate(" ".join(wrds1))
plt.figure(figsize=(13,14))
plt.imshow(wc1,interpolation="bilinear")
plt.axis("off")
plt.title("Coach names- word cloud",color='b')
plt.show()
cup_mask = np.array(Image.open("../input/footballwc/Football.jpg"))
#footballer_mask = np.array(Image.open("../input/mask-image-fifa-cup/footballer.jpg"))
#ball_mask = np.array(Image.open("../input/mask-image-fifa-cup/ball.jpg"))

wc_cup = WordCloud(background_color="white", max_words=2000, mask=cup_mask)
#wc_footballer = WordCloud(background_color="white", max_words=2000, mask=footballer_mask)
#wc_ball = WordCloud(background_color="white", max_words=2000, mask=ball_mask)

winner_text = ' '.join(wc['Winner'].dropna().tolist())

wc_cup.generate(winner_text)

plt.figure(figsize = (21,12))
sns.set_style("whitegrid")

plt.title('Word cloud of the team that have the most wins', fontsize=14)
plt.imshow(wc_cup, interpolation='bilinear')
plt.axis("off")

plt.show()