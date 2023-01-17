import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline



df=pd.read_csv("D:\\E drive\\Thik\\FullData.csv")

df.head(7)
del df['National_Kit']

df.head(12)
plt.figure(figsize=(15,40))

sb.countplot(y=df.Nationality,palette="Set2")
plt.figure(figsize=(10,6))

sb.countplot(x="Age",data=df)
#BEST GOALKEEPERS

a=0.5

b=1.0

c=1.5

d=2.0

#GOALKEPPER CHARECTERISTICS

df['gk_stopper']=(a*df.Dribbling+a*df.Ball_Control+a*df.Aggression+a*df.Stamina+a*df.Strength+a*df.Skill_Moves+a*df.Shot_Power+a*df.Heading+a*df.Agility+a*df.Finishing+b*df.Short_Pass+b*df.Composure+b*df.Speed+b*df.Acceleration+b*df.Balance+b*df.Interceptions+c*df.Long_Pass+c*df.Sliding_Tackle+c*df.Standing_Tackle+c*df.Jumping+d*df.GK_Reflexes+d*df.GK_Positioning+d*df.GK_Diving+d*df.GK_Handling+d*df.GK_Kicking)

df['gk_swepper']=(a*df.Dribbling+a*df.Ball_Control+a*df.Stamina+a*df.Strength+a*df.Shot_Power+a*df.Heading+a*df.Agility+a*df.Finishing+b*df.Aggression+b*df.Skill_Moves+b*df.Composure+b*df.Acceleration+b*df.Balance+c*df.Short_Pass+c*df.Speed+c*df.Long_Pass+c*df.Sliding_Tackle+c*df.Standing_Tackle+c*df.Jumping+c*df.Interceptions+d*df.GK_Reflexes+d*df.GK_Positioning+d*df.GK_Diving+d*df.GK_Handling+d*df.GK_Kicking) 

plt.figure(figsize=(20,12))



sd1= df.sort_values('gk_stopper', ascending=False)[:5]

x1= np.array(list(sd1['Name']))

y1= np.array(list(sd1['gk_stopper']))



sb.barplot(x1,y1, palette= 'Set3')

plt.ylabel('Shot_Stopper_Scorer')
plt.figure(figsize=(28,16))



sd = df.sort_values('gk_swepper', ascending=False)[:10]

x2= np.array(list(sd1['Name']))

y2= np.array(list(sd1['gk_swepper']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('Swepper')
#Defender Charecteristics

df['df_Wing_backs']=(b*df.Dribbling+b*df.Ball_Control+c*df.Aggression+c*df.Stamina+b*df.Strength+b*df.Skill_Moves+b*df.Shot_Power+a*df.Heading+a*df.Agility+b*df.Finishing+c*df.Short_Pass+b*df.Composure+d*df.Speed+c*df.Acceleration+b*df.Balance+c*df.Interceptions+b*df.Long_Pass+c*df.Sliding_Tackle+c*df.Standing_Tackle+b*df.Jumping+b*df.Long_Shots+c*df.Crossing+c*df.Marking)

df['df_centre_backs']=(a*df.Dribbling+a*df.Ball_Control+b*df.Aggression+c*df.Stamina+b*df.Strength+a*df.Skill_Moves+a*df.Shot_Power+c*df.Heading+b*df.Agility+b*df.Finishing+c*df.Short_Pass+b*df.Composure+a*df.Speed+b*df.Acceleration+b*df.Balance+d*df.Interceptions+c*df.Long_Pass+d*df.Sliding_Tackle+d*df.Standing_Tackle+d*df.Jumping+b*df.Long_Shots+a*df.Crossing+d*df.Marking)
#Centre Backs

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'LCB')].sort_values('df_centre_backs', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_centre_backs']))



sb.barplot(x2,y2,palette='Set2')

plt.ylabel('BEST LCB')
plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'RCB')].sort_values('df_centre_backs', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_centre_backs']))



sb.barplot(x2,y2,palette='Set2')

plt.ylabel('BEST RCB')   
#Best Wingbacks

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'LWB') | (df['Club_Position'] == 'LB')].sort_values('df_Wing_backs', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Wing_backs']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST LWB/LB')   
plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'RWB') | (df['Club_Position'] == 'RB')].sort_values('df_Wing_backs', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Wing_backs']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST RWB/RB')   
df['df_Holding_Midfielder']=(a*df.Dribbling+b*df.Ball_Control+a*df.Aggression+c*df.Stamina+c*df.Strength+b*df.Skill_Moves+a*df.Shot_Power+b*df.Heading+a*df.Agility+b*df.Finishing+d*df.Short_Pass+c*df.Composure+b*df.Speed+c*df.Acceleration+c*df.Balance+d*df.Interceptions+d*df.Long_Pass+c*df.Sliding_Tackle+c*df.Standing_Tackle+c*df.Jumping+c*df.Long_Shots+a*df.Crossing+d*df.Marking)

df['df_Playmaker']=(d*df.Dribbling+d*df.Ball_Control+c*df.Aggression+c*df.Stamina+b*df.Strength+d*df.Skill_Moves+c*df.Shot_Power+b*df.Heading+b*df.Agility+c*df.Finishing+d*df.Short_Pass+d*df.Composure+c*df.Speed+c*df.Acceleration+d*df.Balance+a*df.Interceptions+c*df.Long_Pass+a*df.Sliding_Tackle+a*df.Standing_Tackle+c*df.Jumping+c*df.Long_Shots+c*df.Crossing+a*df.Marking)

df['df_Balace_Midfielder']=(c*df.Dribbling+c*df.Ball_Control+b*df.Aggression+c*df.Stamina+c*df.Strength+c*df.Skill_Moves+c*df.Shot_Power+c*df.Heading+b*df.Agility+c*df.Finishing+d*df.Short_Pass+d*df.Composure+b*df.Speed+c*df.Acceleration+d*df.Balance+c*df.Interceptions+c*df.Long_Pass+b*df.Sliding_Tackle+b*df.Standing_Tackle+c*df.Jumping+d*df.Long_Shots+c*df.Crossing+b*df.Marking)



#Best Holding Midfielder

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'CDM') | (df['Club_Position'] == 'LDM') |(df['Club_Position'] == 'RDM')].sort_values('df_Holding_Midfielder', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Holding_Midfielder']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST CDM')   
#Best Playmaker

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'CM') | (df['Club_Position'] == 'LCM')].sort_values('df_Playmaker', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Playmaker']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST CM/LCM')
#Best Balanced Midfielder

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'CM') | (df['Club_Position'] == 'RCM')].sort_values('df_Balace_Midfielder', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Balace_Midfielder']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST CM/RCM')
df['df_Right_Winger']=(d*df.Dribbling+d*df.Ball_Control+c*df.Aggression+c*df.Stamina+b*df.Strength+d*df.Skill_Moves+d*df.Shot_Power+b*df.Heading+c*df.Agility+d*df.Finishing+d*df.Short_Pass+d*df.Composure+c*df.Speed+c*df.Acceleration+d*df.Balance+c*df.Vision+c*df.Weak_foot+a*df.Interceptions+d*df.Long_Pass+a*df.Sliding_Tackle+a*df.Standing_Tackle+c*df.Jumping+c*df.Long_Shots+d*df.Crossing+a*df.Marking+c*df.Attacking_Position+d*df.Curve+c*df.Volleys+d*df.Penalties)

df['df_Left_Winger']=(d*df.Dribbling+d*df.Ball_Control+c*df.Aggression+c*df.Stamina+b*df.Strength+d*df.Skill_Moves+d*df.Shot_Power+b*df.Heading+c*df.Agility+d*df.Finishing+d*df.Short_Pass+d*df.Composure+c*df.Speed+c*df.Acceleration+d*df.Balance+c*df.Vision+c*df.Weak_foot+a*df.Interceptions+d*df.Long_Pass+a*df.Sliding_Tackle+a*df.Standing_Tackle+c*df.Jumping+c*df.Long_Shots+d*df.Crossing+a*df.Marking+c*df.Attacking_Position+d*df.Curve+c*df.Volleys+d*df.Penalties)

df['df_Advance_Striker']=(c*df.Weak_foot+b*df.Vision+b*df.Dribbling+c*df.Ball_Control+d*df.Aggression+c*df.Stamina+c*df.Strength+d*df.Skill_Moves+d*df.Shot_Power+c*df.Heading+d*df.Agility+d*df.Finishing+c*df.Short_Pass+d*df.Composure+d*df.Speed+d*df.Acceleration+c*df.Balance+a*df.Interceptions+b*df.Long_Pass+a*df.Sliding_Tackle+a*df.Standing_Tackle+d*df.Jumping+c*df.Long_Shots+b*df.Crossing+a*df.Marking+d*df.Attacking_Position+b*df.Curve+c*df.Volleys+d*df.Penalties)

plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'RW') | (df['Club_Position'] == 'SS')].sort_values('df_Right_Winger', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Right_Winger']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST RW/SS')
plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'LW') | (df['Club_Position'] == 'SS')].sort_values('df_Left_Winger', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Left_Winger']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST LW/SS')
plt.figure(figsize=(15,8))



sd = df[(df['Club_Position'] == 'ST')].sort_values('df_Advance_Striker', ascending=False)[:5]

x2= np.array(list(sd['Name']))

y2= np.array(list(sd['df_Advance_Striker']))



sb.barplot(x2,y2,palette='Set3')

plt.ylabel('BEST ST')
#Defenders-Thiago Silva,Azipilicuatta,Valencia,Alaba

#Goalkeppers-Manual Nueur

#Midfielders-Autro Vidal,Kan'te,Pogba

#Attackers-Lionel Messi,Christiano Ronaldo,Luis Suarez