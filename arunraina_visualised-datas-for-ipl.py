import pandas as pd
import matplotlib.pyplot as plt
deliveries = pd.read_csv('../input/deliveries.csv')
matches = pd.read_csv('../input/matches.csv')
deliveries
import seaborn as sns 
%matplotlib inline
import numpy as np
matches.describe()
deliveries.describe()
matches[matches.season == 2017].winner.unique()
mat = matches[matches.season == 2017]
sh = len(mat[mat.winner == 'Sunrisers Hyderabad'].winner)
rps = len(mat[mat.winner == 'Rising Pune Supergiant'].winner)
kkr = len(mat[mat.winner == 'Kolkata Knight Riders'].winner)
kxp = len(mat[mat.winner == 'Kings XI Punjab'].winner)
rcb = len(mat[mat.winner == 'Royal Challengers Bangalore'].winner)
mi = len(mat[mat.winner == 'Mumbai Indians'].winner)
dd = len(mat[mat.winner == 'Delhi Daredevils'].winner)
#gl = len(mat[mat.winner == 'Gujarat Lions'].winner)
labels = ['Sunrisers Hyderabad', 'Rising Pune Supergiant',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi Daredevils']
sizes = [sh,rps,kkr,kxp,rcb,mi,dd]
colors = ['orange','purple','blue','yellow','red','cyan','violet']
explode = [0,0,0,0,0.25,0.25,0]
plt.pie(sizes,labels=labels,colors = colors,autopct='%1.1f%%',shadow=True,radius = 2,explode=explode)
plt.title("Winning % of 2017",x=0.59,y=-.7,color = 'green',size = 40)
matches[matches.season == 2014].winner.unique()
mat14 = matches[matches.season == 2014]
rr_14 = len(mat14[mat14.winner == 'Rajasthan Royals'].winner)
csk_14 = len(mat14[mat14.winner == 'Chennai Super Kings'].winner)
kkr_14 = len(mat14[mat14.winner == 'Kolkata Knight Riders'].winner)
kxp_14 = len(mat14[mat14.winner == 'Kings XI Punjab'].winner)
rcb_14 = len(mat14[mat14.winner == 'Royal Challengers Bangalore'].winner)
mi_14 = len(mat14[mat14.winner == 'Mumbai Indians'].winner)
dd_14 = len(mat14[mat14.winner == 'Delhi Daredevils'].winner)
sh_14 = len(mat14[mat14.winner == 'Sunrisers Hyderabad'].winner)
labels1 = ['Sunrisers Hyderabad', 'CHENNAI SUPER KINGS',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi Daredevils']
sizes = [sh_14,csk_14,kkr_14,kxp_14,rcb_14,mi_14,dd_14]
colors = ['orange','purple','blue','yellow','red','cyan','violet']
explode = [0,0,0,0.25,0,0,0.25]
plt.pie(sizes,labels=labels1,colors = colors,autopct='%1.1f%%',shadow=True,radius = 2,explode=explode)
plt.title("Winning Percentage of 2014 IPL",x=0.5,y=-1,size = 40,color = 'green')
#Chose 5 players to display the most Player_of_the_match
c_gayle= len(matches[matches.player_of_match == 'CH Gayle'])
y_sing = len(matches[matches.player_of_match == 'Yuvraj Singh'])
ms_dhoni = len(matches[matches.player_of_match == 'MS Dhoni'])
v_kohli = len(matches[matches.player_of_match == 'V Kohli'])
s_raina = len(matches[matches.player_of_match == 'SK Raina'])
#Most Player_of_the_match among selected 5 players
 #---------#-----#------#-----#
labels = ['CHRIS GAYLE','YUVARAJ SINGH','MS DHONI','VIRAT KOHLI','SURESH RAINA']
sizes = [c_gayle,y_sing,ms_dhoni,v_kohli,s_raina]
plt.pie(sizes,labels=labels,explode = [0.15,0,0,0,0.0],autopct ='%1.1f%%', shadow = True)
plt.title("Most Player of the match award between",color='green',x=.8,y=-0.2)

#sh_l,kkr_l,dd_l,csk_l,kxp_l,rr_l = Total matches they played
sh = matches[matches.team1 == 'Sunrisers Hyderabad'].team1
sh1 = matches[matches.team2 == 'Sunrisers Hyderabad'].team2
sh_l = len(sh)+len(sh1)

kkr = matches[matches.team1 == 'Mumbai Indians'].team1
kkr1 = matches[matches.team2 == 'Mumbai Indians'].team2
kkr_l = len(kkr)+len(kkr1)

mi = matches[matches.team1 == 'Mumbai Indians'].team1
mi1 = matches[matches.team2 == 'Mumbai Indians'].team2
mi_l = len(mi)+len(mi1)

rcb = matches[matches.team1 == 'Royal Challengers Bangalore'].team1
rcb1 = matches[matches.team2 == 'Royal Challengers Bangalore'].team2
rcb_l = len(rcb)+len(rcb1)

csk = matches[matches.team1 == 'Chennai Super Kings'].team1
csk1 = matches[matches.team2 == 'Chennai Super Kings'].team2
csk_l = len(csk)+len(csk1)

kt = matches[matches.team1 == 'Kochi Tuskers Kerala'].team1
kt1 = matches[matches.team2 == 'Kochi Tuskers Kerala'].team2
kt_l = len(kt)+len(kt1)

dd = matches[matches.team1 == 'Delhi Daredevils'].team1
dd1 = matches[matches.team2 == 'Delhi Daredevils'].team2
dd_l = len(sh)+len(sh1)


kxp = matches[matches.team1 == 'Kings XI Punjab'].team1
kxp1 = matches[matches.team2 == 'Kings XI Punjab'].team2
kxp_l = len(kxp)+len(kxp1)


rr = matches[matches.team1 == 'Rajasthan Royals'].team1
rr1 = matches[matches.team2 == 'Rajasthan Royals'].team2
rr_l = len(rr)+len(rr1)

rps = matches[matches.team1 == 'Rising Pune Supergiants'].team1
rps1 = matches[matches.team2 == 'Rising Pune Supergiants'].team2
rps_l = len(rps)+len(rps1)
print("Total Mtches played by teams")
print("CSK = " ,csk_l ,'\n' "RR = ", rr_l,'\n' "DD = " , dd_l,'\n' "RPS = " , rps_l,'\n' "RCB = " , rcb_l,'\n' "MI = " ,mi_l,'\n' "KT = " ,kt_l)
      

rr_w  = len(matches[matches.winner == 'Rajasthan Royals'].winner)
csk_w = len(matches[matches.winner == 'Chennai Super Kings'].winner)
kkr_w = len(matches[matches.winner == 'Kolkata Knight Riders'].winner)
kxp_w= len(matches[matches.winner == 'Kings XI Punjab'].winner)
rcb_w= len(matches[matches.winner == 'Royal Challengers Bangalore'].winner)
mi_w = len(matches[matches.winner == 'Mumbai Indians'].winner)
dd_w = len(matches[matches.winner == 'Delhi Daredevils'].winner)
sh_w = len(matches[matches.winner == 'Sunrisers Hyderabad'].winner)
matches_played = [sh_l,kkr_l,mi_l,rcb_l,csk_l,dd_l,kxp_l,rr_l]
matches_won = [sh_w,kkr_w,mi_w,rcb_w,csk_w,dd_w,kxp_w,rr_w]
total_teams = 8
teams_arnge = np.arange(total_teams)
#width = 0.50

plt.figure(figsize=(10,10))

matches_played = [sh_l,kkr_l,mi_l,rcb_l,csk_l,dd_l,kxp_l,rr_l]
matches_won = [sh_w,kkr_w,mi_w,rcb_w,csk_w,dd_w,kxp_w,rr_w]

Total_played = plt.bar(teams_arnge, matches_played,  color='dodgerblue')#width,
Total_won = plt.bar(teams_arnge, matches_won,  color='Lime')#width,

plt.ylabel('Number of Matches')
plt.xlabel('Teams')
plt.title('Overall performance of the team')
plt.xticks(teams_arnge , ('SRH', 'KKR', 'MI', 'RCB', 'CSK', 'DD', 'KXIP', 'RR'))
plt.yticks(np.arange(0, 200, 5))
plt.legend((Total_played[0],Total_won[0]), ('Total Matches Played', 'Matches won'))


mat_8 = matches[matches.season == 2008]
kkr_8w = len(mat_8[mat_8.winner == 'Kolkata Knight Riders'])
kxp_8w = len(mat_8[mat_8.winner == 'Kings XI Punjab'])
mi_8w = len(mat_8[mat_8.winner == 'Mumbai Indians'])
rr_8w = len(mat_8[mat_8.winner == 'Rajasthan Royals'])
dc_8w = len(mat_8[mat_8.winner == 'Deccan Chargers'])
csk_8w = len(mat_8[mat_8.winner == 'Chennai Super Kings'])
dd_8w = len(mat_8[mat_8.winner == 'Delhi Daredevils'])
rcb_8w = len(mat_8[mat_8.winner == 'Royal Challengers Bangalore'])
rps_8w = len(mat_8[mat_8.winner == 'Rising Pune Supergiants'])
gl_8w = len(mat_8[mat_8.winner == 'Gujarat Lions'])
srh_8w = len(mat_8[mat_8.winner == 'Sunrisers Hyderabad'])
mat_9 = matches[matches.season == 2009]
kkr_9w = len(mat_9[mat_9.winner == 'Kolkata Knight Riders'])
kxp_9w = len(mat_9[mat_9.winner == 'Kings XI Punjab'])
mi_9w = len(mat_9[mat_9.winner == 'Mumbai Indians'])
rr_9w = len(mat_9[mat_9.winner == 'Rajasthan Royals'])
dc_9w = len(mat_9[mat_9.winner == 'Deccan Chargers'])
csk_9w = len(mat_9[mat_9.winner == 'Chennai Super Kings'])
dd_9w = len(mat_9[mat_9.winner == 'Delhi Daredevils'])
rcb_9w = len(mat_9[mat_9.winner == 'Royal Challengers Bangalore'])
rps_9w = len(mat_9[mat_9.winner == 'Rising Pune Supergiants'])
gl_9w = len(mat_9[mat_9.winner == 'Gujarat Lions'])
srh_9w = len(mat_9[mat_9.winner == 'Sunrisers Hyderabad'])
mat_10 = matches[matches.season == 2010]
kkr_10w = len(mat_10[mat_10.winner == 'Kolkata Knight Riders'])
kxp_10w = len(mat_10[mat_10.winner == 'Kings XI Punjab'])
mi_10w = len(mat_10[mat_10.winner == 'Mumbai Indians'])
rr_10w = len(mat_10[mat_10.winner == 'Rajasthan Royals'])
dc_10w = len(mat_10[mat_10.winner == 'Deccan Chargers'])
csk_10w = len(mat_10[mat_10.winner == 'Chennai Super Kings'])
dd_10w = len(mat_10[mat_10.winner == 'Delhi Daredevils'])
rcb_10w = len(mat_10[mat_10.winner == 'Royal Challengers Bangalore'])
rps_10w = len(mat_10[mat_10.winner == 'Rising Pune Supergiants'])
gl_10w = len(mat_10[mat_10.winner == 'Gujarat Lions'])
srh_10w = len(mat_10[mat_10.winner == 'Sunrisers Hyderabad'])
mat_11 = matches[matches.season == 2011]
kkr_11w = len(mat_11[mat_11.winner == 'Kolkata Knight Riders'])
kxp_11w = len(mat_11[mat_11.winner == 'Kings XI Punjab'])
mi_11w = len(mat_11[mat_11.winner == 'Mumbai Indians'])
rr_11w = len(mat_11[mat_11.winner == 'Rajasthan Royals'])
dc_11w = len(mat_11[mat_11.winner == 'Deccan Chargers'])
csk_11w = len(mat_11[mat_11.winner == 'Chennai Super Kings'])
dd_11w = len(mat_11[mat_11.winner == 'Delhi Daredevils'])
rcb_11w = len(mat_11[mat_11.winner == 'Royal Challengers Bangalore'])
kt_11w = len(mat_11[mat_11.winner == 'Kochi Tuskers Kerala'])
rps_11w = len(mat_11[mat_11.winner == 'Pune Warriors'])
gl_11w = len(mat_11[mat_11.winner == 'Gujarat Lions'])
srh_11w = len(mat_11[mat_11.winner == 'Sunrisers Hyderabad'])

mat_12 = matches[matches.season == 2012]
kkr_12w = len(mat_12[mat_12.winner == 'Kolkata Knight Riders'])
kxp_12w = len(mat_12[mat_12.winner == 'Kings XI Punjab'])
mi_12w = len(mat_12[mat_12.winner == 'Mumbai Indians'])
rr_12w = len(mat_12[mat_12.winner == 'Rajasthan Royals'])
dc_12w = len(mat_12[mat_12.winner == 'Deccan Chargers'])
csk_12w = len(mat_12[mat_12.winner == 'Chennai Super Kings'])
dd_12w = len(mat_12[mat_12.winner == 'Delhi Daredevils'])
rcb_12w = len(mat_12[mat_12.winner == 'Royal Challengers Bangalore'])
kt_12w = len(mat_12[mat_12.winner == 'Kochi Tuskers Kerala'])
rps_12w = len(mat_12[mat_12.winner == 'Pune Warriors'])
rps_12w = len(mat_12[mat_12.winner == 'Rising Pune Supergiants'])
gl_12w = len(mat_12[mat_12.winner == 'Gujarat Lions'])
srh_12w = len(mat_12[mat_12.winner == 'Sunrisers Hyderabad'])
mat_13 = matches[matches.season == 2013]
kkr_13w = len(mat_13[mat_13.winner == 'Kolkata Knight Riders'])
kxp_13w = len(mat_13[mat_13.winner == 'Kings XI Punjab'])
mi_13w = len(mat_13[mat_13.winner == 'Mumbai Indians'])
rr_13w = len(mat_13[mat_13.winner == 'Rajasthan Royals'])
dc_13w = len(mat_13[mat_13.winner == 'Deccan Chargers'])
csk_13w = len(mat_13[mat_13.winner == 'Chennai Super Kings'])
dd_13w = len(mat_13[mat_13.winner == 'Delhi Daredevils'])
rcb_13w = len(mat_13[mat_13.winner == 'Royal Challengers Bangalore'])
kt_13w = len(mat_13[mat_13.winner == 'Kochi Tuskers Kerala'])
rps_13w = len(mat_13[mat_13.winner == 'Pune Warriors'])
gl_13w = len(mat_13[mat_13.winner == 'Gujarat Lions'])
srh_13w = len(mat_13[mat_13.winner == 'Sunrisers Hyderabad'])
mat_14 = matches[matches.season == 2014]
kkr_14w = len(mat_14[mat_14.winner == 'Kolkata Knight Riders'])
kxp_14w = len(mat_14[mat_14.winner == 'Kings XI Punjab'])
mi_14w = len(mat_14[mat_14.winner == 'Mumbai Indians'])
rr_14w = len(mat_14[mat_14.winner == 'Rajasthan Royals'])
dc_14w = len(mat_14[mat_14.winner == 'Deccan Chargers'])
csk_14w = len(mat_14[mat_14.winner == 'Chennai Super Kings'])
dd_14w = len(mat_14[mat_14.winner == 'Delhi Daredevils'])
rcb_14w = len(mat_14[mat_14.winner == 'Royal Challengers Bangalore'])
kt_14w = len(mat_14[mat_14.winner == 'Kochi Tuskers Kerala'])
rps_14w = len(mat_14[mat_14.winner == 'Rising Pune Supergiant'])
srh_14w = len(mat_14[mat_14.winner == 'Sunrisers Hyderabad'])
gl_14w = len(mat_14[mat_14.winner == 'Gujarat Lions'])

mat_15 = matches[matches.season == 2015]
kkr_15w = len(mat_15[mat_15.winner == 'Kolkata Knight Riders'])
kxp_15w = len(mat_15[mat_15.winner == 'Kings XI Punjab'])
mi_15w = len(mat_15[mat_15.winner == 'Mumbai Indians'])
rr_15w = len(mat_15[mat_15.winner == 'Rajasthan Royals'])
dc_15w = len(mat_15[mat_15.winner == 'Deccan Chargers'])
csk_15w = len(mat_15[mat_15.winner == 'Chennai Super Kings'])
dd_15w = len(mat_15[mat_15.winner == 'Delhi Daredevils'])
rcb_15w = len(mat_15[mat_15.winner == 'Royal Challengers Bangalore'])
kt_15w = len(mat_15[mat_15.winner == 'Kochi Tuskers Kerala'])
rps_15w = len(mat_15[mat_15.winner == 'Rising Pune Supergiant'])
srh_15w = len(mat_15[mat_15.winner == 'Sunrisers Hyderabad'])
gl_14w = len(mat_15[mat_15.winner == 'Gujarat Lions'])
mat_16 = matches[matches.season == 2016]
kkr_16w = len(mat_16[mat_16.winner == 'Kolkata Knight Riders'])
kxp_16w = len(mat_16[mat_16.winner == 'Kings XI Punjab'])
mi_16w = len(mat_16[mat_16.winner == 'Mumbai Indians'])
rr_16w = len(mat_16[mat_16.winner == 'Rajasthan Royals'])
dc_16w = len(mat_16[mat_16.winner == 'Deccan Chargers'])
csk_16w = len(mat_16[mat_16.winner == 'Chennai Super Kings'])
dd_16w = len(mat_16[mat_16.winner == 'Delhi Daredevils'])
rcb_16w = len(mat_16[mat_16.winner == 'Royal Challengers Bangalore'])
kt_16w = len(mat_16[mat_16.winner == 'Kochi Tuskers Kerala'])
rps_16w = len(mat_16[mat_16.winner == 'Rising Pune Supergiants'])
srh_16w = len(mat_16[mat_16.winner == 'Sunrisers Hyderabad'])
gl_16w = len(mat_16[mat_16.winner == 'Gujarat Lions'])
mat_17 = matches[matches.season == 2017]
kkr_17w = len(mat_17[mat_17.winner == 'Kolkata Knight Riders'])
kxp_17w = len(mat_17[mat_17.winner == 'Kings XI Punjab'])
mi_17w = len(mat_17[mat_17.winner == 'Mumbai Indians'])
rr_17w = len(mat_17[mat_17.winner == 'Rajasthan Royals'])
dc_17w = len(mat_17[mat_17.winner == 'Deccan Chargers'])
csk_17w = len(mat_17[mat_17.winner == 'Chennai Super Kings'])
dd_17w = len(mat_17[mat_17.winner == 'Delhi Daredevils'])
rcb_17w = len(mat_17[mat_17.winner == 'Royal Challengers Bangalore'])
kt_17w = len(mat_17[mat_17.winner == 'Kochi Tuskers Kerala'])
rps_17w = len(mat_17[mat_17.winner == 'Rising Pune Supergiant'])
srh_17w = len(mat_17[mat_17.winner == 'Sunrisers Hyderabad'])
gl_17w = len(mat_17[mat_17.winner == 'Gujarat Lions'])
csk_win = [csk_8w,csk_9w,csk_10w,csk_11w,csk_12w,csk_13w,csk_14w,csk_15w,csk_16w,csk_17w]
kkr_win = [kkr_8w,kkr_9w,kkr_10w,kkr_11w,kkr_12w,kkr_13w,kkr_14w,kkr_15w,kkr_16w,kkr_17w]
dd_win = [dd_8w,dd_9w,dd_10w,dd_11w,dd_12w,dd_13w,dd_14w,dd_15w,dd_16w,dd_17w]
rcb_win = [rcb_8w,rcb_9w,rcb_10w,rcb_11w,rcb_12w,rcb_13w,rcb_14w,rcb_15w,rcb_16w,rcb_17w]
mi_win = [mi_8w,mi_9w,mi_10w,mi_11w,mi_12w,mi_13w,mi_14w,mi_15w,mi_16w,mi_17w]
rps_win = [rps_8w,rps_9w,rps_10w,rps_11w,rps_12w,rps_13w,rps_14w,rps_15w,rps_16w,rps_17w]
kxp_win = [kxp_8w,kxp_9w,kxp_10w,kxp_11w,kxp_12w,kxp_13w,kxp_14w,kxp_15w,kxp_16w,kxp_17w]
rr_win = [rr_8w,rr_9w,rr_10w,rr_11w,rr_12w,rr_13w,rr_14w,rr_15w,rr_16w,rr_17w]
plt.rcParams['figure.figsize'] = 16,11
plt.title("Total Matches Won In Appropriate Seasons",size = 30,color = 'c')
plt.plot(csk_win,color = 'gold',marker = 's',ls = '--',label = 'Chennai Super Kings')
plt.plot(kkr_win,color = 'purple',marker = 's',ls = '--',label = 'Kolkata Night Riders')
plt.plot(dd_win,color = 'blue',marker = 's',ls = '--',label = 'Delhi Daredevils')
plt.plot(rcb_win,color = 'red',marker = 's',ls = '--',label = 'Royal Challengers Bangalore')
plt.plot(mi_win,color = 'cyan',marker = 's',ls = '--',label = 'Mumbai Indians')
plt.plot(rps_win,color = 'violet',marker = 's',ls = '--',label = 'Raising Pune SuperGiants')
plt.plot(kxp_win,color = 'teal',marker =  's',ls = '--',label = 'Kings XI Punjab')
plt.plot(rr_win,color = 'lime',marker = 's',ls = '--',label = 'Rajasthan Royals')
plt.ylabel("Number of matches won",size = 25,color = 'blue')
plt.xlabel("Seasons",size = 25,color = 'blue')

plt.yticks(np.arange(0,15))
years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
plt.xticks(np.arange(0,10),years)

plt.legend(loc = 'upper right')

plt.rcParams['figure.figsize'] = 15,10
sns.set_style("whitegrid")
#plt.rcParams['figure.figsize'] = 15,20
plt.xticks(rotation=-90 , size = 20)
plt.title("BOX PLOT",size = 20)
plt.yticks(size = 20)
plt.xlabel("Winner",size = 40)
plt.ylabel("season",size = 20)
bx = sns.boxplot(data=matches,x = matches.winner,y = matches.season )
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Simple Dashboard -- - - - - - -   ##########


labels2 = ['Sunrisers Hyderabad', 'CHENNAI SUPER KINGS',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi Daredevils']

sizes = [sh_14,csk_14,kkr_14,kxp_14,rcb_14,mi_14,dd_14]

import seaborn as sns
import matplotlib.pylab as pylab
sns.set_style("whitegrid")
dash , axes = plt.subplots(2,2, figsize  =(15,15))
axes[0,0].plot(kkr_win,color = 'purple',marker = 's',ls = '--',label = 'Kolkata Night Riders' )
axes[0,0].plot(csk_win,color = 'gold',marker = 's',ls = '--',label = 'Chennai Super Kings')
axes[0,0].plot(kkr_win,color = 'purple',marker = 's',ls = '--',label = 'Kolkata Night Riders')
axes[0,0].plot(dd_win,color = 'blue',marker = 's',ls = '--',label = 'Delhi Daredevils')
axes[0,0].plot(rcb_win,color = 'red',marker = 's',ls = '--',label = 'Royal Challengers Bangalore')
axes[0,0].plot(mi_win,color = 'cyan',marker = 's',ls = '--',label = 'Mumbai Indians')
axes[0,0].plot(rps_win,color = 'violet',marker = 's',ls = '--',label = 'Raising Pune SuperGiants')
axes[0,0].plot(kxp_win,color = 'teal',marker =  's',ls = '--',label = 'Kings XI Punjab')
axes[0,0].plot(rr_win,color = 'lime',marker = 's',ls = '--',label = 'Rajasthan Royals')

plt.xticks(rotation= -90)
axes[1,0].bar(teams_arnge, matches_played, color='dodgerblue')
axes[1,0].bar(teams_arnge, matches_won,  color='Lime')
axes[0,1].pie(sizes,labels = labels2,colors = colors,autopct='%1.1f%%',shadow=True,radius = 0.7)
bx = sns.boxplot(data=matches,x = matches.winner,y = matches.season ,ax= axes[1,1])


plt.rcParams['figure.figsize'] = 15,5
plt.xticks(rotation = -90)
plt.title("Total Matches Played in cities",size = 30)
sns.countplot(x='venue',data = matches)
plt.xlabel("venue",size = 25,color = 'dodgerblue')
plt.ylabel("Matches Played",size = 25,color = 'dodgerblue')
labels5 = ['Bat','Field']
plt.rcParams['figure.figsize'] = 5,5
sns.countplot(data = matches,x = matches.toss_decision)
matches.season.value_counts()
plt.rcParams['figure.figsize'] = 15,7
sns.countplot(data = matches,x = matches.season)
plt.xlabel("Seasons",size = 25,color = 'dodgerblue')
plt.ylabel("Matches Played",size = 25,color = 'dodgerblue')
plt.title("Total Matches Played In All The Season",size = 30,color = 'c')


%matplotlib inline
plt.rcParams['figure.figsize'] = 20,40
sns.countplot(data = matches,y = matches.player_of_match)
plt.xticks(rotation = -90)
plt.title("Most Player Of The Match Award",size=30,x=0.5,y=-0.2,color = 'fuchsia')
