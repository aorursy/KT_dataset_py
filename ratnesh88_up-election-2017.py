import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
file = "../input/up_res.csv"

data = pd.read_csv(file,sep=",")

data.head()
data.info()
print(data.groupby('party').size())
my_colors = ['orange','navy','g','k','w','yellowgreen','r','c']  



plt.title("Party wise Seats Contested",fontsize=30,color='navy')

plt.xlabel("Party",fontsize=20,color='navy')

plt.ylabel("No. of seats contested",fontsize=20,color='navy')



data['party'].value_counts().plot("bar",figsize=(12,6))
votes_vs_party= data.groupby('party')['votes'].sum()

votes_per_party = np.array(votes_vs_party)



votes_vs_party
party_col=data['party']

vote_col=data['votes']

parties= sorted(set(party_col))

seats_per_party = party_col.value_counts().reindex(votes_vs_party.index)
votes_percentage_party = votes_per_party/votes_per_party.sum()*100



for p,v in zip(parties,votes_percentage_party):

    print(p,':',v,'%')
plt.figure(figsize=(10,10))

plt.title("Vote share per party",fontsize=30,color='navy')

plt.pie(votes_per_party,labels=votes_vs_party.index,shadow=True,autopct='%1.1f%%',colors=my_colors)

plt.show()
index_min_vote= np.where(vote_col==min(data['votes']))

data.iloc[index_min_vote]
index_max_vote= np.where(vote_col==max(data['votes']))

data.iloc[index_max_vote]
data.groupby('seat_allotment').size()
column_x = data.columns[0:len(data.columns) - 1]

column_x
votes_per_party_per_seat= votes_per_party/seats_per_party.values

votes_per_party_per_seat
plt.figure(figsize=(10,10))

plt.title("Vote share of party per contested seat",fontsize=30,color='navy')

plt.pie(votes_per_party_per_seat,labels=votes_vs_party.index,shadow=True,colors=my_colors,autopct='%1.1f%%')

plt.show()
votes_vs_phase = data.groupby('phase')['votes'].sum()

votes_vs_phase
plt.figure(figsize=(10,10))

totalvotes = votes_vs_phase.values.sum()

plt.title("Voting per phase",fontsize=30,color='navy')

plt.pie(votes_vs_phase,labels=votes_vs_phase.index,shadow=True,colors=sorted(my_colors))

plt.show()
l = []

for v in range(1,8):

    votes_vs_party= data[data['phase']==v].groupby('party')['votes'].sum()

    l.append(votes_vs_party.values)

l = np.array(l)

df2 = pd.DataFrame(l, columns= votes_vs_party.index, index=range(1,8))



df2.plot.bar(figsize=(12,8),stacked=True,color=my_colors)

plt.title("Phase wise vote division among party",fontsize=30,color='navy')

plt.xlabel("Phase",fontsize=20,color='navy')

plt.ylabel("Votes",fontsize=20,color='navy')
#Phase BJP votes

bjpdata = data.loc[data['party'] == 'BJP+']

bjpdata[10:20]
phase_by_bjpvote = bjpdata.groupby('phase')['votes'].sum()

phase_by_bjpvote
plt.figure(figsize=(12,6))

plt.title("BJP votes per phase")

plt.bar(phase_by_bjpvote.index,phase_by_bjpvote.values,color='orange')

plt.show()
bjp_voting_percentage_per_phase = phase_by_bjpvote/votes_vs_phase*100

print(bjp_voting_percentage_per_phase)
plt.figure(figsize=(12,6))

totalbjpvotes = votes_vs_phase.values.sum()

plt.title("BJP voting percentage per phase")

plt.bar(bjp_voting_percentage_per_phase.index,bjp_voting_percentage_per_phase.values,color='orange')

plt.show()
maxbjpvote = max(bjpdata['votes'])

index_max_bjpvote= np.where(bjpdata['votes']==maxbjpvote)

bjpdata.iloc[index_max_bjpvote]
minbjpvote = min(bjpdata['votes'])

index_min_bjpvote= np.where(bjpdata['votes']==minbjpvote)

bjpdata.iloc[index_min_bjpvote]
bjpdata_ac =bjpdata.groupby('ac')['votes'].sum()

bjpdata_district =bjpdata.groupby('district')['votes'].sum()



bjpdata_ac.head(4)
data_ac = data.groupby('ac')['votes'].sum()

data_district =data.groupby('district')['votes'].sum()



data_ac.head(4)
bjp_vote_margin = data_ac - bjpdata_ac

bjp_vote_margin.head()
bjp_vote_margin1= bjp_vote_margin.reindex().sort_values(ascending=False)

s=bjp_vote_margin1[0:20].plot("bar",figsize=(12,6),fontsize=12)

s.set_title("Top 20 big wins of BJP+",color='r',fontsize=30)

s.set_xlabel("Assembly name",color='m',fontsize=20)

s.set_ylabel("Vote margins",color='m',fontsize=20)
bjp_vote_margin1= bjp_vote_margin.reindex().sort_values(ascending=True)

s=bjp_vote_margin1[0:20].plot("barh",figsize=(12,6),fontsize=12)

s.set_title("20 low votes of BJP+",color='r',fontsize=30)

s.set_xlabel("Vote margins",color='m',fontsize=20)

s.set_ylabel("Assembly name",color='m',fontsize=20)
bjp_district_vote_margin = data_district-bjpdata_district

bjp_district_vote_margin.head()
bjp_district_vote_margin1= bjp_district_vote_margin.reindex().sort_values(ascending=False)

s=bjp_district_vote_margin1[0:20].plot("bar",figsize=(12,6),fontsize=12)

s.set_title("Top 20 big wins of BJP+ district wise",color='r',fontsize=30)

s.set_xlabel("District name",color='m',fontsize=20)

s.set_ylabel("Vote margins",color='m',fontsize=20)
bjp_district_vote_margin1= bjp_district_vote_margin.reindex().sort_values(ascending=True)

s=bjp_district_vote_margin1[0:20].plot("bar",figsize=(12,6),fontsize=12)

s.set_title("20 Low wins of BJP+ district wise",color='r',fontsize=30)

s.set_xlabel("District name",color='m',fontsize=20)

s.set_ylabel("Vote margins",color='m',fontsize=20)
s = data.groupby('ac_no')['votes'].max()

wins = data[(data['ac_no'].isin(s.index)) & (data['votes'].isin(s.values)) ]

wins.head()

seats_won = wins['party'].value_counts()

seats_won.plot("bar",figsize=(12,10),color=my_colors)

plt.title("Seats won by parties",fontsize=30,color='navy')

plt.xlabel("Phase",fontsize=20,color='navy')

plt.ylabel("Seats",fontsize=20,color='navy')
l = []

for v in range(1,8):

    win_party= wins[wins['phase']==v].groupby('party').size()

    wins_per_party = np.array(win_party.values)

    l.append(wins_per_party)



df2 = pd.DataFrame(l, index=range(1,8))



df2.plot.bar(figsize=(12,8),stacked=True,color=my_colors)

plt.title("Phase wise seats division among winning parties",fontsize=30,color='navy')

plt.legend(votes_vs_party.index.drop("None of the Above"))

plt.xlabel("Phase",fontsize=20,color='navy')

plt.ylabel("Seats",fontsize=20,color='navy')
plt.figure(figsize=(12,10))

plt.scatter(x=seats_per_party.drop("None of the Above"),y=seats_won,s=votes_per_party/10**4,color=my_colors)

plt.title("Election development",fontsize=30,color='navy')

plt.xlabel("Seats contested",fontsize=20,color='navy')

plt.ylabel("Seats won",fontsize=20,color='navy')
datax = data[['candidate','party','phase','votes']][data['party'] !='None of the Above'][data['party'] !='others']

cand_names = datax['candidate']

def surnames(x):

    return x.split()[-1]

datax['surname'] = cand_names.map(surnames)



surnames =datax['surname']

s =surnames.value_counts()[0:20].plot("bar",figsize=(12,6),fontsize=12)

s.set_title("Top castes contested UP election",color='g',fontsize=30)

s.set_xlabel("Caste name",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)
top10_caste = surnames.value_counts()[0:10].index

top10_caste
s1=datax.groupby('party')['surname'].value_counts()

print(s1['BJP+'][0:5])
l=[]

for p in ['BJP+', 'BSP', 'INC', 'Independent', 'RLD', 'SP']:

    caste_per_party = s1[p][top10_caste]

    l.append(caste_per_party.values)

    

df = pd.DataFrame(l,index=['BJP+', 'BSP', 'INC', 'Independent', 'RLD', 'SP'],columns=top10_caste)

s=df.plot(kind="bar",stacked=True,figsize=(12,6),fontsize=12)

s.set_title("Ticket division castwise per party",color='g',fontsize=30)

s.set_xlabel("Party name",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)

s2=datax.groupby('phase')['surname'].value_counts()

l=[]

for p in range(1,8):

    caste_per_party = s2[p][top10_caste]

    l.append(caste_per_party.values)



df = pd.DataFrame(l,index=range(1,8),columns=top10_caste)

s=df.plot(kind="bar",figsize=(12,6),fontsize=12)

s.set_title("Phase wise top castes",color='g',fontsize=30)

s.set_xlabel("Phases",color='b',fontsize=20)

s.set_ylabel("Frequency",color='b',fontsize=20)