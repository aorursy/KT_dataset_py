import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/FullData.csv')

Names = pd.read_csv('../input/PlayerNames.csv')

df.assign(Index=np.nan)

df['Index'] = [v.split('/')[2] for v in Names['url']]

del df['Nationality']

del df['National_Position']

del df['National_Kit']

df.head()
#weigts

a = 0.5

b = 1

c= 2

d = 3



#GoalKeeping Indices

df['gk_Shot_Stopper'] = (b*df.Reactions + b*df.Composure + a*df.Speed + a*df.Strength + c*df.Jumping + b*df.GK_Positioning + c*df.GK_Diving + d*df.GK_Reflexes + b*df.GK_Handling)/(2*a + 4*b + 2*c + 1*d)

df['gk_Sweeper'] = (b*df.Reactions + b*df.Composure + b*df.Speed + a*df.Short_Pass + a*df.Long_Pass + b*df.Jumping + b*df.GK_Positioning + b*df.GK_Diving + d*df.GK_Reflexes + b*df.GK_Handling + d*df.GK_Kicking + c*df.Vision)/(2*a + 4*b + 3*c + 2*d)



#Defending Indices

df['df_Hulk'] = (c*df.Marking + c*df.Sliding_Tackle + c*df.Standing_Tackle + c*df.Aggression + a*df.Reactions + b*df.Interceptions + c*df.Strength)/(5*c + b + a)

df['df_Aerial_Beast'] = (c*df.Marking + b*df.Sliding_Tackle + b*df.Standing_Tackle + c*df.Sliding_Tackle + c*df.Reactions + c*df.Interceptions + b*df.Balance + c*df.Jumping + b*df.Agility + d*df.Heading)/(5*c + 4*b + d)

df['df_Sweeper'] = (c*df.Ball_Control + b*df.Reactions + b*df.Interceptions + d*df.Vision + b*df.Composure + b*df.Short_Pass + b*df.Long_Pass)/(5*b + c + d)

df['df_wb_Wing_Back_Eff'] = (b*df.Ball_Control + a*df.Dribbling + a*df.Marking + c*df.Sliding_Tackle + b*df.Standing_Tackle + c*df.Attacking_Position + d*df.Vision + c*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + d*df.Acceleration +d*df.Speed + c*df.Stamina + a*df.Finishing)/(3*a + 4*b + 4*c + 3*d)



#Midfielding Indices

df['mf_controller'] = (a*df.Weak_foot + c*df.Ball_Control + a*df.Dribbling + a*df.Marking + a*df.Reactions + d*df.Vision + c*df.Composure + d*df.Short_Pass + d*df.Long_Pass)/(2*c + 3*d + 4*a)

df['mf_beast'] = (b*df.Agility + b*df.Balance + b*df.Jumping + c*df.Strength + c*df.Stamina + b*df.Speed + a*df.Acceleration + b*df.Short_Pass + d*df.Aggression + d*df.Reactions + d*df.Marking + c*df.Standing_Tackle + c*df.Sliding_Tackle + d*df.Interceptions)/(1*a + 5*b + 4*c + 4*d)

df['mf_playmaker'] = (b*df.Ball_Control + a*df.Dribbling + a*df.Marking + b*df.Reactions + d*df.Vision + c*df.Crossing + c*df.Short_Pass + c*df.Long_Pass + a*df.Curve + a*df.Long_Shots + c*df.Freekick_Accuracy)/(4*a + 2*b + 4*c + d)

df['mf_attacker'] = (b*df.Ball_Control + c*df.Dribbling + b*df.Vision + b*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + c*df.Agility + a*df.Curve + c*df.Long_Shots + b*df.Freekick_Accuracy + d*df.Finishing)/(a + 6*b + 3*c + d)



#Attackers

df['at_left_wing'] = (c*df.Weak_foot + c*df.Ball_Control + c*df.Dribbling + c*df.Speed + d*df.Acceleration + b*df.Vision + c*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + b*df.Aggression + b*df.Agility + a*df.Curve + c*df.Long_Shots + b*df.Freekick_Accuracy + d*df.Finishing)/(a + 6*b + 6*c + 2*d)

df['at_right_wing'] = (c*df.Weak_foot + c*df.Ball_Control + c*df.Dribbling + c*df.Speed + d*df.Acceleration + b*df.Vision + c*df.Crossing + b*df.Short_Pass + b*df.Long_Pass + b*df.Aggression + b*df.Agility + a*df.Curve + c*df.Long_Shots + b*df.Freekick_Accuracy + d*df.Finishing)/(a + 6*b + 6*c + 2*d)

df['at_striker'] = (b*df.Weak_foot + b*df.Ball_Control + a*df.Vision + b*df.Aggression + b*df.Agility + a*df.Curve + a*df.Long_Shots + d*df.Balance + d*df.Finishing + d*df.Heading + c*df.Jumping + c*df.Dribbling)/(3*a + 4*b + 2*c + 3*d)



df[['Name', 'gk_Shot_Stopper', 'gk_Sweeper', 'df_Hulk', 'df_Aerial_Beast', 'df_Sweeper', 'df_wb_Wing_Back_Eff', 'mf_controller', 'mf_beast', 'mf_playmaker', 'mf_attacker', 'at_left_wing', 'at_right_wing', 'at_striker']].head()
sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15), sharex=False)



# Generate some sequential data

ss = df.sort_values('gk_Shot_Stopper', ascending=False)[:7]

x1 = np.array(list(ss['Name']))

y1 = np.array(list(ss['gk_Shot_Stopper']))

ss = df.sort_values('gk_Sweeper', ascending=False)[:7]

x2 = np.array(list(ss['Name']))

y2 = np.array(list(ss['gk_Sweeper']))



sns.barplot(x1, y1, palette="GnBu_d", ax=ax1)

ax1.set_ylabel("Shot Stopping Score")



sns.barplot(x2, y2, palette="GnBu_d", ax=ax2)

ax2.set_ylabel("Sweeping Score")
sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 21), sharex=False)



# Generate some sequential data

ss = df[(df['Club_Position'] == 'CB') | (df['Club_Position'] == 'LCB') | (df['Club_Position'] == 'RCB')].sort_values('df_Hulk', ascending=False)[:7]

x1 = np.array(list(ss['Name']))

y1 = np.array(list(ss['df_Hulk']))

ss = df[(df['Club_Position'] == 'CB') | (df['Club_Position'] == 'LCB') | (df['Club_Position'] == 'RCB')].sort_values('df_Aerial_Beast', ascending=False)[:7]

x2 = np.array(list(ss['Name']))

y2 = np.array(list(ss['df_Aerial_Beast']))

ss = df[(df['Club_Position'] == 'CB') | (df['Club_Position'] == 'LCB') | (df['Club_Position'] == 'RCB')].sort_values('df_Sweeper', ascending=False)[:7]

x3 = np.array(list(ss['Name']))

y3 = np.array(list(ss['df_Sweeper']))

ss = df[(df['Club_Position'] == 'LWB') | (df['Club_Position'] == 'LB') | (df['Club_Position'] == 'RWB') | (df['Club_Position'] == 'RB')].sort_values('df_wb_Wing_Back_Eff', ascending=False)[:7]

x4 = np.array(list(ss['Name']))

y4 = np.array(list(ss['df_wb_Wing_Back_Eff']))



sns.barplot(x1, y1, palette=sns.cubehelix_palette(7), ax=ax1)

ax1.set_ylabel("Hulk Score")

sns.barplot(x2, y2, palette=sns.cubehelix_palette(7), ax=ax2)

ax2.set_ylabel("Aerial Beast Score")

sns.barplot(x3, y3, palette=sns.cubehelix_palette(7), ax=ax3)

ax3.set_ylabel("Sweeper Score")

sns.barplot(x4, y4, palette=sns.cubehelix_palette(7), ax=ax4)

ax4.set_ylabel("Wing Back Score")



plt.tight_layout(h_pad=5)
sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 21), sharex=False)



# Generate some sequential data

ss = df[(df['Club_Position'] == 'CDM') | (df['Club_Position'] == 'CM') | (df['Club_Position'] == 'LDM') | (df['Club_Position'] == 'RDM')].sort_values('mf_controller', ascending=False)[:7]

x1 = np.array(list(ss['Name']))

y1 = np.array(list(ss['mf_controller']))

ss = df[(df['Club_Position'] == 'CDM') | (df['Club_Position'] == 'CM') | (df['Club_Position'] == 'LDM') | (df['Club_Position'] == 'RDM')].sort_values('mf_beast', ascending=False)[:7]

x2 = np.array(list(ss['Name']))

y2 = np.array(list(ss['mf_beast']))

ss = df[(df['Club_Position'] == 'CM') | (df['Club_Position'] == 'LCM') | (df['Club_Position'] == 'RCM') | (df['Club_Position'] == 'LM') | (df['Club_Position'] == 'RM')].sort_values('mf_playmaker', ascending=False)[:7]

x3 = np.array(list(ss['Name']))

y3 = np.array(list(ss['mf_playmaker']))

ss = df[(df['Club_Position'] == 'CM') | (df['Club_Position'] == 'LCM') | (df['Club_Position'] == 'RCM') | (df['Club_Position'] == 'LM') | (df['Club_Position'] == 'RM')].sort_values('mf_attacker', ascending=False)[:7]

x4 = np.array(list(ss['Name']))

y4 = np.array(list(ss['mf_attacker']))



sns.barplot(x1, y1, palette=sns.color_palette("RdBu", n_colors=7), ax=ax1)

ax1.set_ylabel("Controller Score")

sns.barplot(x2, y2, palette=sns.color_palette("RdBu", n_colors=7), ax=ax2)

ax2.set_ylabel("Beast Score")

sns.barplot(x3, y3, palette=sns.color_palette("RdBu", n_colors=7), ax=ax3)

ax3.set_ylabel("PlayMaker Score")

sns.barplot(x4, y4, palette=sns.color_palette("RdBu", n_colors=7), ax=ax4)

ax4.set_ylabel("Attacker Score")



plt.tight_layout(h_pad=5)
sns.set(style="white", context="talk")



# Set up the matplotlib figure

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 17), sharex=False)



# Generate some sequential data

ss = df[(df['Club_Position'] == 'LW') | (df['Club_Position'] == 'LM') | (df['Club_Position'] == 'LS')].sort_values('at_left_wing', ascending=False)[:7]

x1 = np.array(list(ss['Name']))

y1 = np.array(list(ss['at_left_wing']))

ss = df[(df['Club_Position'] == 'RW') | (df['Club_Position'] == 'RM') | (df['Club_Position'] == 'RS')].sort_values('at_right_wing', ascending=False)[:7]

x2 = np.array(list(ss['Name']))

y2 = np.array(list(ss['at_right_wing']))

ss = df[(df['Club_Position'] == 'ST') | (df['Club_Position'] == 'LS') | (df['Club_Position'] == 'RS') | (df['Club_Position'] == 'CF')].sort_values('at_striker', ascending=False)[:7]

x3 = np.array(list(ss['Name']))

y3 = np.array(list(ss['at_striker']))



sns.barplot(x1, y1, palette=sns.color_palette('RdGy', n_colors=7), ax=ax1)

ax1.set_ylabel("Left Wing")

sns.barplot(x2, y2, palette=sns.color_palette("RdGy", n_colors=7), ax=ax2)

ax2.set_ylabel("Right Wing")

sns.barplot(x3, y3, palette=sns.color_palette("RdGy", n_colors=7), ax=ax3)

ax3.set_ylabel("Striker")



plt.tight_layout(h_pad=5)