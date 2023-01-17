import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df_electrol_data14 = pd.read_csv("../input/LS2014Electors.csv")

df_electrol_data09 = pd.read_csv("../input/LS2009Electors.csv")

df_candidate_data14 = pd.read_csv("../input/LS2014Candidate.csv")

df_candidate_data09 = pd.read_csv("../input/LS2009Candidate.csv")

df_candidate_data09.head()

df_candidate_data14.head()
total_electors09=df_electrol_data09["Total_Electors"].sum()

print ("There are a total of ",+total_electors09 ,"electors in India")
total_electors=df_electrol_data14["Total_Electors"].sum()

print ("There are a total of ",+total_electors ,"electors in India")
total_voters09=df_electrol_data09["Total voters"].sum()

print("There are a total of ",+total_voters09 ,"voters in India")
total_voters=df_electrol_data14["Total voters"].sum()

print("There are a total of ",+total_voters ,"voters in India")
total_turnout09 = round(total_voters09/total_electors09*100,2)

print("Total Turnout in 2009 is ",+total_turnout09,"%")
total_turnout = round(total_voters/total_electors*100,2)

print("Total Turnout in 2014 is ",+total_turnout,"%")
candidate_sex = df_candidate_data09["Candidate Sex"].value_counts()

candidate_sex
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.pie(df_candidate_data09[(df_candidate_data09["Party Abbreviation"]=='INC')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)



fig = plt.gcf() 

fig.suptitle("Candidates Gender Distribution in 2009 - INC vs BJP", fontsize=14) 

ax = fig.gca() 

label = ax.annotate("INC", xy=(-1.1,-1), fontsize=30, ha="center",va="center")

ax.axis('off')

ax.set_aspect('equal')

ax.autoscale_view()



plt.subplot(1,2,2)

plt.pie(df_candidate_data09[(df_candidate_data09["Party Abbreviation"]=='BJP')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)

fig = plt.gcf() 

ax = fig.gca() 

label = ax.annotate("BJP", xy=(-1.1,-1), fontsize=30, ha="center",va="center")

ax.axis('off')

ax.set_aspect('equal')

ax.autoscale_view()

plt.show();
candidate_sex = df_candidate_data14["Candidate Sex"].value_counts()

candidate_sex
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='INC')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)



fig = plt.gcf() 

fig.suptitle("Candidates Gender Distribution in 2014 - INC vs BJP", fontsize=14) 

ax = fig.gca() 

label = ax.annotate("INC", xy=(-1.1,-1), fontsize=30, ha="center",va="center")

ax.axis('off')

ax.set_aspect('equal')

ax.autoscale_view()



plt.subplot(1,2,2)

plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='BJP')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)

fig = plt.gcf() 

ax = fig.gca() 

label = ax.annotate("BJP", xy=(-1.1,-1), fontsize=30, ha="center",va="center")

ax.axis('off')

ax.set_aspect('equal')

ax.autoscale_view()

plt.show();
df_womenwinners09 = df_candidate_data09[(df_candidate_data09['Position']==1)&(df_candidate_data09["Candidate Sex"]=="F")]

 

ax = df_womenwinners09["Party Abbreviation"].value_counts().plot(kind="pie",radius=2,autopct='%1.1f%%', startangle=90)

x = df_womenwinners09["Party Abbreviation"].value_counts()

x
df_womenwinners14 = df_candidate_data14[(df_candidate_data14['Position']==1)&(df_candidate_data14["Candidate Sex"]=="F")]

ax1 = df_womenwinners14["Party Abbreviation"].value_counts().plot(kind="pie",radius=2,autopct='%1.1f%%', startangle=90)

x = df_womenwinners14["Party Abbreviation"].value_counts()

x
Age09=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)]['Candidate Age'].tolist()

Age14=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)]['Candidate Age'].tolist()

bins = np.linspace(20, 90, 10)

plt.hist([Age09, Age14], bins, label=['2009', '2014'])



plt.legend(loc='upper right')

plt.xlabel('Age Of winners in years')

plt.ylabel('Total Number of winners')

plt.title('Distribution of Age of the winners')

plt.show()



df_candidate_data09["Alliance"] = df_candidate_data09["Party Abbreviation"]



df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["INC","AITC","DMK","NCP","NC","JMM","MUL","VCK","KEC(M)","AIMIM"],value="UPA")

df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["BJP","JD(U)","SHS","RLD","SAD","TRS","AGP","INLD"],value="NDA")

df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["CPM","CPI","RSP","AIFB","BSP","BJD","ADMK","TDP","JD(S)","MDMK","HJS","PMK"],value="Third Front")

df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["SP","RJD","LJP"],value="Fourth Front")

df_candidate_data09["Alliance"] = df_candidate_data09["Alliance"].replace(to_replace =["AUDF","JKM(P)","NPF","BOPF","SWP","BKA","SDF","IND","JKN","HJCBL","BVA","JVN","JVM"],value="Others")                                                
df_candidate_data14["Alliance"] = df_candidate_data14["Party Abbreviation"]



df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['INC','NCP', 'RJD', 'DMK', 'IUML', 'JMM','JD(s)','KC(M)','RLD','RSP','CMP(J)','KC(J)','PPI','MD'],value="UPA")

df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['BJP','SHS', 'LJP', 'SAD', 'RLSP', 'AD','PMK','NPP','AINRC','NPF','RPI(A)','BPF','JD(U)','SDF','NDPP','MNF','RIDALOS','KMDK','IJK','PNK','JSP','GJM','MGP','GFP','GVP','AJSU','IPFT','MPP','KPP','JKPC','KC(T)','BDJS','AGP','JSS','PPA','UDP','HSPDP','PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF'],value="NDA")



df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['YSRCP',"AITC",'AAAP',"BJD","ADMK",'IND', 'AIUDF', 'BLSP', 'JKPDP',"CPM","TRS","TDP","SP", 'JD(S)', 'INLD', 'CPI', 'AIMIM', 'KEC(M)','SWP', 'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 'BVA', 'HJCBL', 'JVM','MDMK'],value="Others")                            

Age09UPA=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)&(df_candidate_data09.Alliance=="UPA")]['Candidate Age'].tolist()

Age09NDA=df_candidate_data09[(df_candidate_data09.Position==1) & (df_candidate_data09.Year==2009)&(df_candidate_data09.Alliance=="NDA")]['Candidate Age'].tolist()

bins = np.linspace(20, 90, 10)

plt.hist([Age09UPA, Age09NDA], bins, label=['UPA', 'NDA'])

plt.legend(loc='upper right')

plt.xlabel('Age Of winners in years')

plt.ylabel('Total Number of winners')

plt.title('Alliance wise Distribution of Age of the winners in 2009')

plt.show()









Age14UPA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="UPA")]['Candidate Age'].tolist()

Age14NDA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="NDA")]['Candidate Age'].tolist()

bins = np.linspace(20, 90, 10)

plt.hist([Age14UPA, Age14NDA], bins, label=['UPA', 'NDA'])

plt.legend(loc='upper right')

plt.xlabel('Age Of winners in years')

plt.ylabel('Total Number of winners')

plt.title('Alliance wise Distribution of Age of the winners in 2014')

plt.show()
df_winners09 = df_candidate_data09[df_candidate_data09['Position']==1]

DF09 = df_winners09['Party Abbreviation'].value_counts().head(10)

DF09
df_winners09 = df_candidate_data09[df_candidate_data09['Position']==1]

DF09 = df_winners09['Party Abbreviation'].value_counts().head().to_dict()

S09 = sum(df_winners09['Party Abbreviation'].value_counts().tolist())

DF09['Other Regional Parties'] = S09 - sum(df_winners09['Party Abbreviation'].value_counts().head().tolist())

fig = plt.figure()



ax09 = fig.add_axes([0, 0,.5,.5], aspect=1)

colors = ["#264CE4","#FF5106","#E426A4","#44A122","#F2EC3A","#C96F58"]

ax09.pie(DF09.values(),labels=DF09.keys(),autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2,colors = colors)

ax09.set_title("2009",loc="center",fontdict={'fontsize':20},position=(0.5,1.55))

plt.show()

df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]

DF14 = df_winners14['Party Abbreviation'].value_counts().head(10)

DF14
df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]

DF14 = df_winners14['Party Abbreviation'].value_counts().head().to_dict()

S14 = sum(df_winners14['Party Abbreviation'].value_counts().tolist())

DF14['Other Regional Parties'] = S14 - sum(df_winners14['Party Abbreviation'].value_counts().head().tolist())

fig = plt.figure()



ax14 = fig.add_axes([0, 0,.5,.5], aspect=1)

colors = ["#FF5106","#264CE4","#E426A4","#44A122","#F2EC3A","#C96F58"]

ax14.pie(DF14.values(),labels=DF14.keys(),autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2,colors=colors)

ax14.set_title("2014",loc="center",fontdict={'fontsize':20},position=(0.5,1.55))

plt.show()

votespartywise09 = df_candidate_data09.groupby('Party Abbreviation')['Total Votes Polled'].sum()

x09 = votespartywise09.sort_values(ascending=False)[:10].plot(kind="bar")

x09.set_xlabel('Party Abbrevations')

x09.set_ylabel('Votes in Million(100)')

votespartywise09.sort_values(ascending=False)[:10]
votespartywise14 = df_candidate_data14.groupby('Party Abbreviation')['Total Votes Polled'].sum()

x14 = votespartywise14.sort_values(ascending=False)[:10].plot(kind="bar")

x14.set_xlabel('Party Abbrevations')

x14.set_ylabel('Votes in Million(100)')

votespartywise14.sort_values(ascending=False)[:10]
pollper = df_electrol_data09.groupby("STATE").mean()

LS09 = pollper[['POLL PERCENTAGE']].round(1).sort_values('POLL PERCENTAGE',ascending=False)

ax1 =LS09['POLL PERCENTAGE'].plot(kind='bar',figsize=(20, 15))

for p in ax1.patches:

    ax1.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+2),fontsize=12)

pollper14 = df_electrol_data14.groupby("STATE").mean()

LS14 = pollper14[['POLL PERCENTAGE']].round(1).sort_values('POLL PERCENTAGE',ascending=False)

ax14 =LS14['POLL PERCENTAGE'].plot(kind='bar',figsize=(20, 15))

for p in ax14.patches:

    ax14.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+2),fontsize=12)

df_electrol_data14["STATE"] = df_electrol_data14["STATE"].replace(to_replace = ["Odisha"],value="Orissa")

df_electrol_data14["STATE"] = df_electrol_data14["STATE"].replace(to_replace = ["Chhattisgarh"],value="Chattisgarh")
pollper = df_electrol_data09.groupby("STATE").mean()

LS09 = pollper[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()

Y09=[2009 for i in range(35)]

S09=list(LS09['POLL PERCENTAGE'].keys())

P09=list(LS09['POLL PERCENTAGE'].values())



pollper14 = df_electrol_data14.groupby("STATE").mean()

LS14 = pollper14[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()

Y14=[2014 for i in range(35)]

S14=list(LS14['POLL PERCENTAGE'].keys())

P14=list(LS14['POLL PERCENTAGE'].values())





Data = {'YEAR':Y09+Y14,'STATE':S09+S14,'Poll_Percentage':P09+P14}

DF = pd.DataFrame(data=Data)

ax = plt.subplots(figsize=(6, 20))

sns.barplot(x=DF.Poll_Percentage,y=DF.STATE,hue=DF.YEAR)
SeatsWin = df_candidate_data09[(df_candidate_data09.Position==1)].groupby(['Alliance'])['Position'].sum()

SeatsWin

SeatsWin.plot(kind ="pie",autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2)
SeatsWin14 = df_candidate_data14[(df_candidate_data14.Position==1)].groupby(['Alliance'])['Position'].sum()



SeatsWin14.plot(kind ="pie",autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2)
s09 = df_candidate_data09[df_candidate_data09["Position"]==1].groupby("State name")["Alliance"].value_counts()

alliance09 = df_candidate_data09[df_candidate_data09["Position"]==1]["Alliance"].unique()
l = []

for v in ["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',

       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',

       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',

       'Meghalaya', 'Mizoram', 'Nagaland', 'Orissa', 'Punjab',

       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',

       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',

       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',

       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry']:

         win_party09 = s09[v][alliance09]

         l.append(win_party09.values)

            

df = pd.DataFrame(l,index=["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',

       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',

       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',

       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',

       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',

       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',

       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',

       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry'],columns=alliance09)

s=df.plot(kind="bar",stacked=True,figsize=(18,9),fontsize=15)

s.set_title("State wise seats won per Alliance (2009)",color='g',fontsize=30)

s.set_xlabel("Staes",color='b',fontsize=20)

s.set_ylabel("No. of seats",color='b',fontsize=20)
s14 = df_candidate_data14[df_candidate_data14["Position"]==1].groupby("State name")["Alliance"].value_counts()

alliance14 = df_candidate_data14[df_candidate_data14["Position"]==1]["Alliance"].unique()
l = []

for v in ["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',

       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',

       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',

       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',

       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',

       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',

       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',

       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry']:

         win_party = s14[v][alliance14]

         l.append(win_party.values)

            

df = pd.DataFrame(l,index=["Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Goa',

       'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',

       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',

       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',

       'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh',

       'West Bengal', 'Chattisgarh', 'Jharkhand', 'Uttarakhand',

       'Andaman & Nicobar Islands', 'Chandigarh', 'Dadra & Nagar Haveli',

       'Daman & Diu', 'NCT OF Delhi', 'Lakshadweep', 'Puducherry'],columns=alliance14)

s=df.plot(kind="bar",stacked=True,figsize=(18,9),fontsize=15)

s.set_title("Statewise seats won per Alliances (2014)",color='g',fontsize=30)

s.set_xlabel("States",color='b',fontsize=20)

s.set_ylabel("Seats",color='b',fontsize=20)