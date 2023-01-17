# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





match_data = pd.read_csv('../input/ipl-auction-and-ipl-dataset/matches.csv')

top_bat = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Top_100_batsman.xlsx')

top_ball = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Top_100_bowlers.xlsx')

a = pd.read_excel('../input/ipl-auction-and-ipl-dataset/Auction.xlsx')
match_data.head(5)
a.head(5)
#https://stackoverflow.com/questions/41192997/separate-out-and-keep-duplicate-categorical-data-using-seaborn-barplot/41193156#41193156



a1_color = [ '#8C411E','#F97324','#F9CD05','#EC1C24','#00ADEF','#FB2897','#004BA0','#00008B','#9E3495','#C0D6EB']



plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

a_r =a.nlargest(10, ['PRICE(In crore Indian Rupees) '])

plt.bar(np.arange(10),a_r.iloc[0:11,3],color=a1_color,align='center')

plt.xticks(np.arange(10),a_r.iloc[0:11,1],rotation=90,fontsize=15)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel('PRICE (In crore Indian Rupees)',fontsize=15)

plt.title("Top 10 buys in IPL history",fontsize=20)

plt.show()
#To get n largest values from dataframe got help from:-

#https://www.geeksforgeeks.org/get-n-largest-values-from-a-particular-column-in-pandas-dataframe/



for i in range(0,56,8):

    q = a.iloc[i:i+8,:].nlargest(5,['PRICE(In crore Indian Rupees) '])

    plt.figure(figsize=(17,6))

    plt.yticks(fontsize=15)

    plt.bar(q.loc[:,'PLAYER'],q.loc[:,'PRICE(In crore Indian Rupees) '])

    plt.xlabel("PLAYER",fontsize = 15)

    plt.ylabel("PRICE(In crore Indian Rupees) ",fontsize = 15)

    plt.xticks(rotation=90,fontsize = 20)

    plt.title(f"Top 5 buys for Year {q.iloc[2,4]}",fontsize = 20)

    plt.show()
a1 = dict(match_data['venue'].value_counts())

a1_venue = list(a1.keys())

a1_matches = list(a1.values())

a1_color = [ '#00008B','#00ADEF','#8C411E','#F97324','#EC1C24','#004BA0','#00008B','#F9CD05','#9E3495','#C0D6EB']



#ploting help got from :-

#https://matplotlib.org/gallery/index.html

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(a1_venue[0:11],a1_matches[0:11],color = a1_color)

plt.xlabel("Venues",fontsize=15)

plt.ylabel("Matches",fontsize=15)

plt.xticks(rotation = 90 ,fontsize=15)

plt.title("Top 10 Venues",fontsize=20)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
top_bat_r =top_bat.nlargest(10, ['Runs'])



plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(top_bat_r.loc[0:11,'PLAYER'],top_bat_r.loc[0:11,'Runs'],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize = 15)

plt.ylabel("RUNS",fontsize = 15)

plt.xticks(rotation = 90 ,fontsize = 15)

plt.title("Top 10 Run Scorer in IPL history",fontsize = 20)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()

sixes = top_bat.nlargest(10, ['6s'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(sixes.iloc[0:11,1],sixes.iloc[0:11,13],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("SIXES",fontsize=15)

plt.title("Most six hitters in IPL history",fontsize=20)

plt.xticks(rotation =90,fontsize=15 )

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()

fours = top_bat.nlargest(10, ['4s'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(fours.iloc[0:11,1],fours.iloc[0:11,12],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("FOURS",fontsize=15)

plt.title("Most four hitters in IPL history",fontsize=20)

plt.xticks(rotation = 90 ,fontsize=15)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
s_r = top_bat.nlargest(10, ['SR'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(s_r.iloc[0:11,1],s_r.iloc[0:11,9],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("STRIKE RATE",fontsize=15)

plt.title("Best batting strike raters in IPL history",fontsize=20)

plt.xticks(rotation = 45,fontsize=15 )

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
top_ball_r = top_ball.nlargest(10, ['Wkts'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(top_ball_r.loc[0:11,'PLAYER'],top_ball_r.loc[0:11,'Wkts'],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("WICKETS",fontsize=15)

plt.title("Top 10 Wicket takers in IPL history",fontsize=20)

plt.xticks(rotation = 90,fontsize=15 )

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
s_r = top_ball.nsmallest(10, ['Avg'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(s_r.iloc[0:11,1],s_r.iloc[0:11,8],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("AVERAGE",fontsize=15)

plt.title("Best bowling average in IPL history",fontsize=20)

plt.xticks(rotation = 90 ,fontsize=15)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
s_r = top_ball.nsmallest(10, ['Econ'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(s_r.iloc[0:11,1],s_r.iloc[0:11,9],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("ECONOMY",fontsize=15)

plt.title("Best bowling economy in IPL history",fontsize=20)

plt.xticks(rotation = 90 ,fontsize=15)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
s_r = top_ball.nsmallest(10, ['SR'])

plt.figure(figsize=(17,6))

plt.yticks(fontsize=15)

plt.bar(s_r.iloc[0:11,1],s_r.iloc[0:11,10],color = a1_color)

#,tick_label=a1_matches[0:11]

plt.xlabel("PLAYER",fontsize=15)

plt.ylabel("STRIKE RATE",fontsize=15)

plt.title("Best bowling strike rate in IPL history",fontsize=20)

plt.xticks(rotation = 90 ,fontsize=15)

plt.grid(True , linestyle = '--',linewidth = 0.3)

plt.show()
#Got help from sahib virji notebook:-

#https://www.kaggle.com/sahib12/ipl-match-analysis



def comparator(team1):

    teams=list(match_data.team1.unique())# you can take team2 here also

    teams.remove(team1)

    opponents=teams.copy()

    mt1=match_data[((match_data['team1']==team1)|(match_data['team2']==team1))]

    g1 = []

    g2 = []

    g3 = []



    for i in opponents:

        mask = (((mt1['team1']==i)|(mt1['team2']==i)))&((mt1['team1']==team1)|(mt1['team2']==team1))

        mt2 = mt1.loc[ mask,'winner'].value_counts()

        g1.append(i)

        try:

            g2.append(mt2.loc[i])

        except Exception as e:

            g2.append(0)

        try:

            g3.append(mt2.loc[team1])



        except Exception as e:

            g3.append(0)

    plt.figure(figsize=(17,6))

    plt.grid(True, linestyle="-.", linewidth="0.3")

    plt.bar([a - 0.4 for a in range(len(g1))], g2, width=0.4, align='edge',color='#F9CD05')

    plt.bar([a for a in range(len(g1))], g3, label=team1, width=0.4, align='edge',color = '#9E3495')

    plt.title(f"{team1} VS all teams in IPL history",fontsize=20)

    plt.xlabel("",fontsize=15)

    plt.ylabel("Wins",fontsize=15)

    plt.legend(loc='best')

    plt.xticks(range(len(g1)),g1,rotation=90,fontsize=15)

    plt.yticks(range(20),fontsize=15)

    plt.show()





teams=list(match_data.team1.unique())

for j in teams:

    comparator(j)
#To change the font size on a pie chart got help from:-

#https://stackoverflow.com/questions/7082345/how-to-set-the-labels-size-on-a-pie-chart-in-python



c = match_data.loc[(match_data['venue'] == 'Rajiv Gandhi International Stadium, Uppal')]

c_w = c[c['win_by_runs']>0]

s2 = [len(c_w),len(c) - len(c_w)]

l=['Batting first','Batting Second']

plt.figure(figsize=(15,5))

plt.pie(s2,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#F9CD05','#EC1C24'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title("Rajiv Gandhi International Stadium, Uppal",fontsize=20)

plt.show()
m = match_data.loc[(match_data['venue'] == 'Punjab Cricket Association IS Bindra Stadium, Mohali')]

m_w = m[m['win_by_runs'] > 0]

s3 = [len(m_w),len(m) - len(m_w)]

plt.figure(figsize=(15,5))

plt.pie(s3,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#9E3495','#F97324'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('Punjab Cricket Association IS Bindra Stadium, Mohali',fontsize=20)

plt.show()

d = match_data.loc[(match_data['venue']=='Feroz Shah Kotla') ]

d_w=d[d['win_by_runs']>0]

s1 =[len(d_w),len(d)-len(d_w)]

plt.figure(figsize=(15,5))

plt.pie(s1,labels=l,startangle=90,shadow=1,autopct='%1.2f%%',colors=['#F9CD05','#9E3495'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title("Feroz Shah Kotla",fontsize=20)

plt.show()
w = match_data.loc[(match_data['venue'] == 'Wankhede Stadium')]

w_w = w[w['win_by_runs']>0]

s4 = [len(w_w),len(w) - len(w_w)]

plt.figure(figsize=(15,5))

plt.pie(s4,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#EC1C24','#00ADEF'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('Wankhede Stadium',fontsize=20)

plt.show()
r = match_data.loc[(match_data['venue'] == 'MA Chidambaram Stadium, Chepauk')]

r_w = r[r['win_by_runs']>0]

s5 = [len(r_w),len(r)-len(r_w)]

plt.figure(figsize=(15,5))

plt.pie(s5,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#F97324','#00ADEF'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('MA Chidambaram Stadium, Chepauk',fontsize=20)

plt.show()
e = match_data[(match_data['venue'] == 'Eden Gardens')]

e_w = e[e['win_by_runs']>0]

s6 = [len(e_w),len(e)-len(e_w)]

plt.figure(figsize = (15,5))

plt.pie(s6,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#EC1C24','#F97324'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('Eden Gardens',fontsize=20)

plt.show()
j = match_data[(match_data['venue'] == 'Sawai Mansingh Stadium')]

j_w = j[j['win_by_runs']>0]

s7 = [len(j_w),len(j)-len(j_w)]

plt.figure(figsize=(15,5))

plt.pie(s7,labels=l,startangle=100,shadow=1, autopct='%1.2f%%',colors=['#9E3495','#00ADEF'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('Sawai Mansingh Stadium',fontsize=20)

plt.show()
t = match_data[(match_data['venue'] == 'M. Chinnaswamy Stadium')]

t_w = t[t['win_by_runs']>0]

s8 = [len(t_w),len(t)-len(t_w)]

plt.figure(figsize=(15,5))

plt.pie(s8,labels=l,startangle=100,shadow=1,autopct='%1.2f%%',colors=['#9E3495','#F97324'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title('M. Chinnaswamy Stadium',fontsize=20)

plt.show()
d = match_data.loc[(match_data['toss_decision']=='bat') ]

d_w=d[d['win_by_runs']>0]

s1 =[len(d_w),len(d)-len(d_w)]

l=['Batting first','Batting Second']

plt.figure(figsize=(15,5))

plt.pie(s1,labels=l,startangle=90,shadow=1,autopct='%1.2f%%',colors=['#F9CD05','#9E3495'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

#explode=(0,0.1) ,

plt.title("All time win percentage in IPL history on basis of batting first and batting second",fontsize=20)

plt.show()
v = match_data.loc[(match_data['toss_decision'] =='bat') ]

sv =[len(v),len(match_data['toss_decision'])-len(v)]

l=['Batting','Bowling']

plt.figure(figsize=(15,5))

plt.pie(sv,labels=l,startangle=90,shadow=1,autopct='%1.2f%%',colors=['#F9CD05','#EC1C24'],wedgeprops={'width': 0.6},textprops={'fontsize': 13})

plt.title("All time percentage in IPL history of what teams did after wining toss",fontsize = 20)

plt.show()
