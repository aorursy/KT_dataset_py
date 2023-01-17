import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("E:/DSE_LERNING/pythonlec_training/IN_CLASS/MIMI PROJECT/fifa.csv")
data.head(2)
data.shape
def Value_float(Value):

    if isinstance(Value,str):

        out = Value.replace('€', '')

        if 'M' in out:

            out = float(out.replace('M', ''))*1000000

        elif 'K' in Value:

            out = float(out.replace('K', ''))*1000

        return float(out)
data['Value'] = data['Value'].apply(Value_float)

data['Wage'] = data['Wage'].apply(Value_float)

data['Release Clause'] = data['Release Clause'].apply(Value_float)
data[['Value','Wage','Release Clause']].head()
data.isna().sum()
data['Joined']=data['Joined'].fillna('0')
def str4(x):

    if isinstance(x,str):

        return int(x[-4:])

data['Joined'] = data['Joined'].apply(str4)
data["Joined"].head()
data['Contract Valid Until']= pd.to_datetime(data['Contract Valid Until'])
data['Contract Valid Until'].head()
def llbs(x):

    l=[]

    if isinstance(x,str):

        l = x.split("lbs")

        i = l[0]

        return float(i)

data['Weight'] = data['Weight'].apply(llbs)  
data['Weight'].head()
def feet(x):

    l=[]

    if isinstance(x,str):

        l = x.split("'")

        i = int(l[0])

        j = int(l[1])

        f = ((i*12)+j)/12

        return f

data['Height'] = data['Height'].apply(feet) 
data['Height'].head()
df = data
df.drop(['Preferred Foot','Weak Foot','Loaned From'],axis=1,inplace=True)
df['Weight'].fillna((df['Weight'].mean()), inplace = True) 

df['International Reputation'].fillna(df['International Reputation'].mean(), inplace = True) 

df['Skill Moves'].fillna(df['Skill Moves'].mean(), inplace = True) 

df['Work Rate'].fillna('Not available', inplace = True) 

df['Body Type'].fillna('Not available', inplace = True) 

df['Position'].fillna('Not available', inplace = True) 

df['Joined'].fillna('Not available', inplace = True) 

df['Contract Valid Until'].fillna('Not available', inplace = True) 

df['Crossing'].fillna(df['Crossing'].mean(), inplace = True) 

df['Finishing'].fillna(df['Finishing'].mean(), inplace = True) 

df['HeadingAccuracy'].fillna(df['HeadingAccuracy'].mean(), inplace = True) 

df['ShortPassing'].fillna(df['ShortPassing'].mean(), inplace = True) 

df['Volleys'].fillna(df['Volleys'].mean(), inplace = True) 

df['Dribbling'].fillna(df['Dribbling'].mean(), inplace = True) 

df['Curve'].fillna(df['FKAccuracy'].mean(), inplace = True) 

df['FKAccuracy'].fillna(df['FKAccuracy'].mean(), inplace = True) 

df['LongPassing'].fillna(df['LongPassing'].mean(), inplace = True) 

df['Joined'].fillna(df['Joined'].mean(), inplace = True) 

df['BallControl'].fillna(df['BallControl'].mean(), inplace = True) 

df['Acceleration'].fillna(df['Acceleration'].mean(), inplace = True) 

df['SprintSpeed'].fillna(df['SprintSpeed'].mean(), inplace = True) 

df['Agility'].fillna(df['Agility'].mean(), inplace = True) 

df['Reactions'].fillna(df['Reactions'].mean(), inplace = True) 

df['Balance'].fillna(df['Balance'].mean(), inplace = True) 

df['ShotPower'].fillna(df['ShotPower'].mean(), inplace = True) 

df['Jumping'].fillna(df['Jumping'].mean(), inplace = True) 

df['Stamina'].fillna(df['Stamina'].mean(), inplace = True) 

df['Strength'].fillna(df['Strength'].mean(), inplace = True) 

df['LongShots'].fillna(df['LongShots'].mean(), inplace = True) 

df['Aggression'].fillna(df['Aggression'].mean(), inplace = True) 

df['Interceptions'].fillna(df['Interceptions'].mean(), inplace = True) 

df['SprintSpeed'].fillna(df['Positioning'].mean(), inplace = True) 

df['Agility'].fillna(df['Vision'].mean(), inplace = True) 

df['Penalties'].fillna(df['Penalties'].mean(), inplace = True) 

df['Composure'].fillna(df['Composure'].mean(), inplace = True) 

df['Marking'].fillna(df['Marking'].mean(), inplace = True) 

df['SlidingTackle'].fillna(df['SlidingTackle'].mean(), inplace = True) 

df['StandingTackle'].fillna(df['StandingTackle'].mean(), inplace = True) 

df['GKDiving'].fillna(df['GKDiving'].mean(), inplace = True) 

df['Stamina'].fillna(df['Stamina'].mean(), inplace = True) 

df['GKHandling'].fillna(df['GKHandling'].mean(), inplace = True) 

df['GKKicking'].fillna(df['GKKicking'].mean(), inplace = True) 

df['GKPositioning'].fillna(df['GKPositioning'].mean(), inplace = True) 

df['GKReflexes'].fillna(df['GKReflexes'].mean(), inplace = True) 

df['Release Clause'].fillna(df['Release Clause'].mean(), inplace = True) 

df['Positioning'].fillna(df['Positioning'].mean(), inplace = True) 

df['Vision'].fillna(df['Vision'].mean(), inplace = True) 
plt.figure(figsize=(10,7))

plt.hist(df['Overall'], edgecolor='black')
a = ["Overall", "Value", "Wage", "International Reputation", "Height", "Weight", "Release Clause"]

sns.pairplot(df[a])
df.sort_values(by='Overall', ascending=False )

df["Contract Valid Until"]
def clean_cont(x):

    l =[]

    if isinstance(x,str):

        l=x.split("-")

        d=l[0]

        return d
data["Contract Year"] = data["Contract Valid Until"].apply(clean_cont)

data["Contract Year"]
def Value_float(Value):

    if isinstance(Value,str):

        out = Value.replace('€', '')

        if 'M' in out:

            out = float(out.replace('M', ''))*1000000

        elif 'K' in Value:

            out = float(out.replace('K', ''))*1000

        return float(out)

data['Value'] = data['Value'].apply(Value_float)

data['Wage'] = data['Wage'].apply(Value_float)

data['Release Clause'] = data['Release Clause'].apply(Value_float)
total = data.sort_values(by = "Overall",ascending = False)[data["Contract Year"]=="2020"].head(20)

total
total["Wage"].mean()
total["Age"].mean()
total.corr()
sns.heatmap(total.corr())
total[['Overall','Value']].corr()
a = data["Position"].drop_duplicates()
a.count()
data["Position"]
data.sort_values(by = "Position")
l = list(data["Position"].unique())

l
data[data["Position"]=="RF"].head()
data[data["Position"]=="ST"].head()
data[data["Position"]=="LW"].head()
data[data["Position"]=="GK"].head()
data[data["Position"]=="RCM"].head()
data[data["Position"]=="LF"].head()
data[data["Position"]=="RS"].head()
data[data["Position"]=="RCB"].head()
data[data["Position"]=="LCM"].head()
data[data["Position"]=="CB"].head()
data[data["Position"]=="LDM"].head()
data[data["Position"]=="CAM"].head()
data[data["Position"]=="CDM"].head()
data[data["Position"]=="LS"].head()
data[data["Position"]=="LCB"].head()
data[data["Position"]=="RM"].head()
data[data["Position"]=="LAM"].head()
data[data["Position"]=="LM"].head()
data[data["Position"]=="LB"].head()
data[data["Position"]=="RDM"].head()
data[data["Position"]=="RW"].head()
data[data["Position"]=="CM"].head()
data[data["Position"]=="RB"].head()
data[data["Position"]=="RAM"].head()
data[data["Position"]=="CF"].head()
data[data["Position"]=="RWB"].head()
data[data["Position"]=="LWB"].head()
dd = data["ID"]
l = []

for i in dd:

    if (data[data['ID']==i]['Position'].count()>=2):

        l.append(i)

if l==[]:

    print('NO such a players appearing in more than one table')

else:

    print('YES, there are players appearing in more than one table',set(l))
gk=data[(data['Position']=='GK')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

lb=data[(LB['Position']=='LB')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

cb=data[(CB['Position']=='CB')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

rb=data[(RB['Position']=='RB')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

lm=data[(LM['Position']=='LM')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

rm=data[(RM['Position']=='RM')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

cdm=data[(CDM['Position']=='CDM')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

cam=data[(CAM['Position']=='CAM')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

lw=data[(LW['Position']=='LW')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

rw=data[(RW['Position']=='RW')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(2)

#st=data[(ST['Position']=='ST')&(data['Contract Year']<=2020)][['Name','Overall','Position','Potential','Value','Release Clause']].sort_values(by='Overall',ascending=False).head(1)