# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
% matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print("setup complete")

# Any results you write to the current directory are saved as output.
#load dataset in kernal 
df=pd.read_csv("../input/fifa18_clean.csv")

#analyze first five row of data set
df.head()

#delete the column club logo and flag
del df['Club Logo']
del df ['Flag']

df.head() #upadated dataset
df.head()
df.rename(columns = {'GK reflexes' : 'GK_reflexes','Free kick accuracy': 'Free_kick_accuracy'},inplace=True)
df.rename(columns = {'GK diving': 'GK_diving','GK handling' : 'GK_handling', 'GK kicking' : 'GK_kicking','GK positioning' : 'GK_positioning', 'Long passing' : 'Long_passing', 'Short passing': 'Short_passing', 'Sprint speed' : 'Sprint_speed' },inplace=True)
df.head()
plt.figure (figsize = (15,32))
sns.countplot (y= df.Nationality, palette= "colorblind") #plot the Nationality of players on y axis 
plt.figure (figsize = (15,6))
sns.countplot (x= df.Age, palette= "colorblind") #plot the Nationality of players on x axis 
#weightage
a = 0.5
b = 1
c = 2
d = 3
names = df.columns.values 
names
#listing the goal keeper characterstics
#ability to stop the goal
df['gk_shot_stopper'] = (b*df.Reactions + b*df.Composure + a*df.Sprint_speed + a*df.Strength + c*df.Jumping + b*df.GK_positioning + c*df.GK_diving + d*df.GK_reflexes + b*df.GK_handling) / (2*a + 4*b + 2*c + 1*d)
df['gk_sweeper'] = (b*df.Reactions + b*df.Composure + b*df.Sprint_speed + a*df.Short_passing + a*df.Long_passing + b*df.Jumping + b*df.GK_positioning + b*df.GK_diving + d*df.GK_reflexes + b*df.GK_handling + d*df.GK_kicking + c*df.Vision ) /  (2*a + 7*b + 1*c + 1*d)
df.head()
plt.figure(figsize= (15,6))
data1 = df.sort_values ('gk_shot_stopper',ascending = False)[:5]
x1 = np.array(list(data1['Name']))
y1 = np.array (list(data1['gk_shot_stopper']))

sns.barplot(x1,y1,palette = "colorblind")
plt.xlabel('Player Names')
plt.ylabel('Shot stopping score')

plt.figure(figsize= (15,6))
data2 = df.sort_values ('gk_sweeper',ascending = False)[:5]
x1 = np.array(list(data1['Name']))
y1 = np.array (list(data1['gk_shot_stopper']))

sns.barplot(x1,y1,palette = "colorblind")
plt.xlabel('Player Names')
plt.ylabel('Shot stopping score')

#rename column names
df.rename(columns = {'Ball control' : 'Ball_control','Heading accuracy' : 'Heading_accuracy','Long shots' : 'Long_shots','Shot power' : 'Shot_power', 'Sliding tackle' : 'Sliding_tackle', 'Preferred Positions' : 'Preferred_Positions'},inplace=True)
df.rename(columns = {'Standing tackle' : 'Standing_tackle'},inplace=True)
df.head()
names = df.columns.values 
names
#listing the defender's characterstics
#centre back abilites
df['df_centre_backs'] = (d*df.Reactions + c*df.Interceptions + d*df.Sliding_tackle + d*df.Standing_tackle + b*df.Vision + b*df.Composure + b*df.Crossing + a*df.Short_passing + b*df.Long_passing + c*df.Acceleration + b*df.Sprint_speed 
                         + d*df.Stamina + d*df.Jumping + d*df.Heading_accuracy + b*df.Long_shots + d*df.Marking + c*df.Aggression) / (6*b + 3*c + 7*d)
df['df_wb_wing_backs'] = (b*df.Ball_control + a*df.Dribbling + a*df.Marking + d*df.Sliding_tackle + d*df.Standing_tackle + a*df.Positioning + c*df.Vision + c*df.Crossing + b*df.Short_passing + c*df.Long_passing + d*df.Acceleration + d*df.Sprint_speed
                         + c*df.Stamina + a*df.Finishing) /  (4*a + 2*b + 3*c + 4*d)
#Left centre back
plt.figure(figsize= (15,6))
data3 = df[(df['Preferred_Positions'] == 'LB')].sort_values ('df_centre_backs',ascending = False)[:5]
x2 = np.array(list(data3['Name']))
y2 = np.array (list(data3['df_centre_backs']))

sns.barplot(x2,y2,palette = "Blues_d")
plt.xlabel('Player Names')
plt.ylabel('Left Centre Back Score')


#Right Centre back
plt.figure(figsize= (15,6))
data4 = df[(df['Preferred_Positions'] == 'RB')].sort_values ('df_centre_backs',ascending = False)[:5]
x2 = np.array(list(data4['Name']))
y2 = np.array (list(data4['df_centre_backs']))

sns.barplot(x2,y2,palette = "Blues_d")
plt.xlabel('Player Names')
plt.ylabel('Right centre back Score')


#Left wing back
plt.figure(figsize= (15,6))
data5 = df[(df['Preferred_Positions'] == 'LWB') | (df['Preferred_Positions'] == 'LB')].sort_values ('df_wb_wing_backs',ascending = False)[:5]
x2 = np.array(list(data5['Name']))
y2 = np.array (list(data5['df_wb_wing_backs']))

sns.barplot(x2,y2,palette = "Blues_d")
plt.xlabel('Player Names')
plt.ylabel('Left wing back Score')
#right wing back
plt.figure(figsize= (15,6))
data5 = df[(df['Preferred_Positions'] == 'RWB') | (df['Preferred_Positions'] == 'RB')].sort_values ('df_wb_wing_backs',ascending = False)[:5]
x2 = np.array(list(data5['Name']))
y2 = np.array (list(data5['df_wb_wing_backs']))

sns.barplot(x2,y2,palette = "Blues_d")
plt.xlabel('Player Names')
plt.ylabel('Right wing back Score')
names = df.columns.values 
names
df['mf_controller'] = (d*df.Ball_control + a*df.Dribbling + a*df.Marking + a*df.Reactions + c*df.Vision + c*df.Composure + d*df.Short_passing + d*df.Long_passing) / (3*a + 1*b + 2*c + 1*d)
df['mf_playmaker'] = (d*df.Ball_control + d*df.Dribbling + a*df.Marking + d*df.Reactions + d*df.Vision + c*df.Positioning + c*df.Crossing + d*df.Short_passing + c*df.Long_passing + c*df.Curve + b*df.Long_shots 
                    + c*df.Free_kick_accuracy ) / (1*a + 1*b + 5*c + 5*d)
df['mf_beast'] = (d*df.Agility + c*df.Balance + b*df.Jumping + c*df.Strength + d*df.Stamina + a*df.Sprint_speed + c*df.Acceleration + d*df.Short_passing + c*df.Aggression + d*df.Reactions + b*df.Marking 
                    + b*df.Standing_tackle + b*df.Sliding_tackle ) / (1*a + 2*b + 4*c + 4*d)
#plot controller
plt.figure(figsize= (15,6))
data5 = df[(df['Preferred_Positions'] == 'CM') | (df['Preferred_Positions'] == 'LM')].sort_values('mf_controller', ascending=False)[:5]
x3 = np.array(list(data5['Name']))
y3 = np.array(list(data5['mf_controller']))
sns.barplot(x3, y3, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("Controller Score")
#plot playmaker
plt.figure(figsize= (15,6))
data6 = df[(df['Preferred_Positions'] == 'CAM') | (df['Preferred_Positions'] == 'CM')].sort_values('mf_playmaker', ascending=False)[:5]
x3 = np.array(list(data6['Name']))
y3 = np.array(list(data6['mf_playmaker']))
sns.barplot(x3, y3, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("Playmaker Score")
#plot beast
plt.figure(figsize= (15,6))
data6 = df[(df['Preferred_Positions'] == 'RCM') | (df['Preferred_Positions'] == 'RM')].sort_values('mf_playmaker', ascending=False)[:5]
x3 = np.array(list(data6['Name']))
y3 = np.array(list(data6['mf_beast']))
sns.barplot(x3, y3, palette=sns.diverging_palette(145, 280, s=85, l=25, n=5))
plt.ylabel("beast Score")
df['att_left_wing'] = (c*df.Ball_control + c*df.Dribbling + c*df.Sprint_speed + d*df.Acceleration + b*df.Vision + c*df.Crossing + b*df.Short_passing + b*df.Long_passing + b*df.Aggression + b*df.Agility + a*df.Curve + c*df.Long_shots + b*df.Free_kick_accuracy + d*df.Finishing)/(a + 6*b + 6*c + 2*d)
df['att_right_wing'] = (c*df.Ball_control + c*df.Dribbling + c*df.Sprint_speed + d*df.Acceleration + b*df.Vision + c*df.Crossing + b*df.Short_passing + b*df.Long_passing + b*df.Aggression + b*df.Agility + a*df.Curve + c*df.Long_shots + b*df.Free_kick_accuracy + d*df.Finishing)/(a + 6*b + 6*c + 2*d)
df['att_striker'] = (b*df.Ball_control + a*df.Vision + b*df.Aggression + b*df.Agility + a*df.Curve + a*df.Long_shots + d*df.Balance + d*df.Finishing + d*df.Heading_accuracy + c*df.Jumping + c*df.Dribbling)/(3*a + 4*b + 2*c + 3*d)
#left wing attacker
plt.figure(figsize=(15,6))
ss = df[(df['Preferred_Positions'] == 'LW') | (df['Preferred_Positions'] == 'LM') | (df['Preferred_Positions'] == 'LS')].sort_values('att_left_wing', ascending=False)[:5]
x1 = np.array(list(ss['Name']))
y1 = np.array(list(ss['att_left_wing']))
sns.barplot(x1, y1, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Left Wing")
#plot right wing attacker
plt.figure(figsize=(15,6))
ss = df[(df['Preferred_Positions'] == 'RW') | (df['Preferred_Positions'] == 'RM') | (df['Preferred_Positions'] == 'RS')].sort_values('att_right_wing', ascending=False)[:5]
x2 = np.array(list(ss['Name']))
y2 = np.array(list(ss['att_right_wing']))
sns.barplot(x2, y2, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Right Wing")
#plot striker
plt.figure(figsize=(15,6))

ss = df[(df['Preferred_Positions'] == 'ST') | (df['Preferred_Positions'] == 'LS') | (df['Preferred_Positions'] == 'RS') | (df['Preferred_Positions'] == 'CF')].sort_values('att_striker', ascending=False)[:5]
x3 = np.array(list(ss['Name']))
y3 = np.array(list(ss['att_striker']))
sns.barplot(x3, y3, palette=sns.diverging_palette(255, 133, l=60, n=5, center="dark"))
plt.ylabel("Striker")
