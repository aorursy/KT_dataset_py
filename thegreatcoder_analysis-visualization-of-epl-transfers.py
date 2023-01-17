#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")
#Importing the file
file = "../input/english-premier-league-transfers-20192020/english_premier_league.csv"
#Reading the file in pandas
df = pd.read_csv(file)
#Showing the first 5 rows
df.head()
#Finding the total amount of money spent in EPL Transfers 2019/20
z = df['fee_cleaned'].sum()
print("A total of £{} million was spent in the EPL Transfers 2019/20 seaon".format(z))
#To shoe the matplotlib in the notebook itself
%matplotlib inline
#Setting the size of the plot
plt.rcParams['figure.figsize']=30,15
#Finding the Number of Players sold Based on Position
a = df['position']
av = df['position'].value_counts()
a_dict = list(dict.fromkeys(a))
sns.countplot(x = "position",
              data = df,
              palette="autumn",
              order = df['position'].value_counts().index)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(fontsize = 30,rotation = 60)
plt.yticks(fontsize = 30)
plt.xlabel("Position",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Number of players sold based on position",fontsize = 50)
print(av)
#Finding the Number of Players sold Based on Fee
b = df['fee_cleaned']
bv = df['fee_cleaned'].value_counts()
b_dict = list(dict.fromkeys(b))
plt.plot(bv,c ='Magenta' , ls='', marker = 'o', ms=15)
print(bv)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel("Fee in £ (in million)",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Number of players sold based on fee",fontsize = 50)
#Finding the Number of Players sold Based on Age
c = df['age']
cv = c.value_counts()
c_dict = list(dict.fromkeys(c))
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
sns.countplot(x = "age",
              data = df,
              palette= "Spectral",)
plt.show
plt.xticks(fontsize = 30,rotation = 0)
plt.yticks(fontsize = 30)
plt.xlabel("Age",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Number of players sold based on age",fontsize = 50)
cv.head()
#Finding the Number of Players sold Based on the club that bought players
d = df['club_name']
dv = df['club_name'].value_counts()
d_dict = list(dict.fromkeys(d))
df.head()
sns.countplot(x = 'club_name',data = df,
              palette = 'Blues_d',
             order = df['club_name'].value_counts().index)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.show
plt.xticks(fontsize = 30,rotation = 65)
plt.yticks(fontsize = 30)
plt.xlabel("Club Name",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Club that bought players",fontsize = 50)
#Finding the Number of Players sold Based on Transfer movement
e = df['transfer_movement']
ev = df['transfer_movement'].value_counts()
e_dict = list(dict.fromkeys(c))
ev.plot(kind = 'bar',color = ['red','green'])
print(ev)
plt.show
plt.xticks(fontsize = 30,rotation = 0)
plt.yticks(fontsize = 30)
plt.xlabel("Transfer movement",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Transfer movement",fontsize = 50)
#Finding the Number of Players sold Based on the club that sold players
f = df['club_involved_name'][:20]
fv = df['club_involved_name'].value_counts()[:20]
f_dict = list(dict.fromkeys(f))
sns.countplot(x = "club_involved_name",
              data = df,
              palette="hot",
              order = df['club_involved_name'].value_counts()[:20].index)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.show
plt.xticks(fontsize = 30,rotation = 65)
plt.yticks(fontsize = 30)
plt.xlabel("Club Name",fontsize = 35)
plt.ylabel("Number of players",fontsize = 35)
plt.title("Club that sold players",fontsize = 50)
#Finding how many players were sold above £40 million
fdf = df[df['fee_cleaned']>=40]
fdf.shape
#The name,fee,position and club that bought players above £40 million
adc = list(zip(fdf.player_name, fdf.fee_cleaned,fdf.position,fdf.club_name))
adc
#Total money spent by Manchester City
mcdf = df[df['club_name']=='Manchester City']
mcf = mcdf['fee_cleaned'].sum()
print('Manchester city spent £{} million'.format(mcf))
#Total money spent on Centre-Forwards
cdf = df[df['position']=='Centre-Forward']
cf = cdf['fee_cleaned'].sum()
print(' £{} million was spent on centre-forwards'.format(cf))
#Total money spent on Players of age 22
adf = df[df['age']==22]
af = adf['fee_cleaned'].sum()
print(' £{} million was spent on players of age 22'.format(af))
pdf = df[df['fee_cleaned']>=0].groupby(df['position']).sum()
pdf = pdf[['fee_cleaned']]
pdf
adf = df[df['fee_cleaned']>=0].groupby(df['age']).sum()
adf = adf[['fee_cleaned']]
adf
