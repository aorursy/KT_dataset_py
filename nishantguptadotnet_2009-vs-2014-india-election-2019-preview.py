# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import * # For global titile using fig.suptitle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading 2009 candidate dataset
LS09Cand = pd.read_csv('../input/LS2009Candidate.csv')
print(LS09Cand.shape)
LS09Cand.head()
# Reading 2014 candidate dataset
LS14Cand = pd.read_csv('../input/LS2014Candidate.csv')
print(LS14Cand.shape)
LS14Cand.head()
#Merging 2009 & 2014 Candidate datasets on one another
LS0914Cand = pd.concat([LS09Cand,LS14Cand])
print(LS0914Cand.shape)
LS0914Cand.head()
#Checking all political parties abbreviation so that we can make it conscise by including alliance & significant parties
LS0914Cand['Party Abbreviation'].unique()
#Introducing Alliance column for optimized substitution of Winning Party Abbreviation column
LS0914Cand['Alliance']=LS0914Cand['Party Abbreviation']
LS0914Cand['Alliance']=LS0914Cand['Alliance'].replace(to_replace=['INC','NCP', 'RJD', 'DMK', 'IUML', 'JMM','JD(s)','KC(M)','RLD','RSP','CMP(J)','KC(J)','PPI','MD'],value='UPA')
LS0914Cand['Alliance']=LS0914Cand['Alliance'].replace(to_replace=['BJP','SS', 'LJP', 'SAD', 'RLSP', 'AD','PMK','NPP','AINRC','NPF','RPI(A)','BPF','JD(U)','SDF','NDPP','MNF','RIDALOS','KMDK','IJK','PNK','JSP','GJM','MGP','GFP','GVP','AJSU','IPFT','MPP','KPP','JKPC','KC(T)','BDJS','AGP','JSS','PPA','UDP','HSPDP','PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF'],value='NDA')
LS0914Cand['Alliance']=LS0914Cand['Alliance'].replace(to_replace=['YSRCP','AAAP', 'IND', 'AIUDF', 'BLSP', 'JKPDP', 'JD(S)', 'INLD', 'CPI', 'AIMIM', 'KEC(M)','SWP', 'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 'BVA', 'HJCBL', 'JVM','MDMK'],value='Others')
LS0914Cand
# Winning seats distribution by Major Political Parties & Alliances for 2009 & 2014
SeatsWin = LS0914Cand[(LS0914Cand.Position==1)].groupby(['Alliance','Year'])['Position'].sum().reset_index().pivot(index='Alliance', columns='Year',values='Position').reset_index().fillna(0).sort_values([2014,2009], ascending=False).reset_index(drop = True)
# Removing Index Name
#SeatsWin = pd.DataFrame(data=SeatsWin.values,columns=['Alliance','2009','2014'])
print(SeatsWin['Alliance'].unique())
SeatsWin
#plotting pie chart
plt.pie(SeatsWin[2009])
#plt.show()
# add a circle at the center
my_circle=plt.Circle( (0,0), 0.7, color='white')
#fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle)

label = ax.annotate("2009", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show()
# White circle with label at center
fig, ax = plt.subplots()

ax = fig.add_subplot(111)

x = 0
y = 0

circle = plt.Circle((x, y), radius=1,color='red')

ax.add_patch(circle)

label = ax.annotate("2009", xy=(x, y), fontsize=30, ha="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show()

# Seats won by Alliances and Major Political Parties

colors  = ("orange", "green", "red", "cyan", "brown", "grey", "blue", "indigo", "beige", "yellow","cadetblue","khaki")
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.pie(SeatsWin[2009], labels=SeatsWin['Alliance'], colors=colors,autopct='%1.1f%%')
my_circle1=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
fig.suptitle("Winning Percentages by Alliances and Major Political Parties", fontsize=14) # Adding supertitle with pyplot import
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle1)

label = ax.annotate("2009", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()



#plt.figure(figsize=(20,10))
plt.subplot(1,2,2)
plt.pie(SeatsWin[2014], labels=SeatsWin['Alliance'], colors=colors,autopct='%1.1f%%')
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("2014", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();


## function to add value label to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
        
        
CatWin = LS0914Cand[(LS0914Cand.Position==1)].groupby(['Candidate Category','Year'])['Position'].sum().reset_index().pivot(index='Candidate Category', columns='Year',values='Position').reset_index().fillna(0).sort_values([2014,2009], ascending=False).reset_index(drop = True)
#print(CatWin['Candidate Category'].unique())
#CatWin


nx = CatWin.plot(kind='bar', title ="Winning Category", figsize=(15, 10), legend=True, fontsize=12)
nx.set_xlabel("Candidate Category", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)

# Modifying Axis Labels
labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0] = 'GEN'
labels[1]= 'SC'
labels[2]='ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)
    

CatAlliance09 = LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)].groupby(['Alliance','Candidate Category'])['Position'].sum().unstack().reset_index().fillna(0)
CatAlliance14 = LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)].groupby(['Alliance','Candidate Category'])['Position'].sum().unstack().reset_index().fillna(0)

nx = CatAlliance09.plot(kind='bar', title ="2009 Winning Category", figsize=(15, 10), legend=True, fontsize=12)
nx.set_xlabel("Candidate Category", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)

# Modifying Axis Labels
labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0:11] = CatAlliance09['Alliance']
#labels[1]= 'SC'
#labels[2]='ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)

nx = CatAlliance14.plot(kind='bar', title ="2014 Winning Category", figsize=(15, 10), legend=True, fontsize=12)
nx.set_xlabel("Candidate Category", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)

# Modifying Axis Labels
labels = [item.get_text() for item in nx.get_xticklabels()]
labels[0:11] = CatAlliance14['Alliance']
#labels[1]= 'SC'
#labels[2]='ST'
nx.set_xticklabels(labels)

annot_plot(nx,0.05,5)
plt.style.use('seaborn-deep')
Age09=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)]['Candidate Age'].tolist()
Age14=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)]['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([Age09, Age14], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Distribution of Age of the winners')
plt.show()
# Age Distribution of Winning Candidates in 2009 & 2014 for NDA & UPA in India Elections

plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.style.use('seaborn-deep')

Age09UPA=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='UPA')]['Candidate Age'].tolist()
Age14UPA=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='UPA')]['Candidate Age'].tolist()
Age09NDA=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='NDA')]['Candidate Age'].tolist()
Age14NDA=LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='NDA')]['Candidate Age'].tolist()

bins = np.linspace(20, 90, 10)
plt.hist([Age09NDA, Age14NDA], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of NDA winners in years')
plt.ylabel('Total Number of NDA winners')
plt.title('Distribution of Age of NDA winners')


plt.subplot(1,2,2)
bins = np.linspace(20, 90, 10)
plt.hist([Age09UPA, Age14UPA], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of UPA winners in years')
plt.ylabel('Total Number of UPA winners')
plt.title('Distribution of Age of UPA winners')

plt.show();
# Gender Distribution of Winning Candidates in 2009 & 2014 India Elections
colors = ['#0000CD','#CD3333']
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle1=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() 
fig.suptitle("Gender Distribution in 2009 & 2014 India Elections", fontsize=14) # Adding supertitle with pyplot import
ax = fig.gca() 
ax.add_patch(my_circle1)
label = ax.annotate("2009", xy=(0, 0), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("2014", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();
# Gender Distribution of Winning Candidates in 2009 - NDA vs UPA in India Elections
colors = ['#0000CD','#CD3333']
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='NDA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle1=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() 
fig.suptitle("Gender Distribution in 2009 - NDA vs UPA", fontsize=14) # Adding supertitle with pyplot import
ax = fig.gca() 
ax.add_patch(my_circle1)
label = ax.annotate("NDA", xy=(0, 0), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='UPA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("UPA", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();
# Gender Distribution of Winning Candidates in 2014 - NDA vs UPA in India Elections

colors = ['#0000CD','#CD3333']
plt.figure(figsize=(10,5))
plt.title('2014')
plt.subplot(1,2,1)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='NDA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle1=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() 
fig.suptitle("Gender Distribution in 2014 - NDA vs UPA", fontsize=14) # Adding supertitle with pyplot import
ax = fig.gca() 
ax.add_patch(my_circle1)
label = ax.annotate("NDA", xy=(0, 0), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='UPA')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%',colors=colors, startangle=90)
my_circle2=plt.Circle( (0,0), 0.7, color='white')
fig = plt.gcf() #gcf means get current figure
ax = fig.gca() # gca means get current axis
ax.add_patch(my_circle2)

label = ax.annotate("UPA", xy=(0, 0), fontsize=30, ha="center",va="center")

ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.show();
# Distribution of Winning Seats across states in 2009 & 2014 for NDA & UPA in India Elections

color  = ("green","orange")

State09UPA=pd.DataFrame(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='UPA')]['State name'].value_counts())
State14UPA=pd.DataFrame(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='UPA')]['State name'].value_counts())
State09NDA=pd.DataFrame(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2009)& (LS0914Cand.Alliance=='NDA')]['State name'].value_counts())
State14NDA=pd.DataFrame(LS0914Cand[(LS0914Cand.Position==1) & (LS0914Cand.Year==2014)& (LS0914Cand.Alliance=='NDA')]['State name'].value_counts())


State09 = pd.concat([State09UPA, State09NDA], axis=1, sort=False).fillna(0)
State09.columns = ['UPA','NDA']

State14 = pd.concat([State14UPA, State14NDA], axis=1, sort=False).fillna(0)
State14.columns = ['UPA','NDA']


nx = State09.plot(kind='bar',    # Plot a bar chart
            legend=True,    # Turn the Legend off
            title = "NDA vs UPA across States of India (In 2009)",      
            width=0.75,      # Set bar width as 75% of space available
            figsize=(15,5),  # Set area size (width,height) of plot in inches
            colors= color)
nx.set_xlabel("Election States", fontsize=12)
nx.set_ylabel("Seats Won", fontsize=12)
annot_plot(nx,0.05,0.5);

kx = State14.plot(kind='bar',    # Plot a bar chart
            legend=True,    # Turn the Legend off
            title = "NDA vs UPA across States of India (In 2014)",      
            width=0.75,      # Set bar width as 75% of space available
            figsize=(15,5),  # Set area size (width,height) of plot in inches
            colors= color)
kx.set_xlabel("Election States", fontsize=12)
kx.set_ylabel("Seats Won", fontsize=12)
annot_plot(kx,0.05,0.5);


# Reading 2009 Electors dataset
LS09Elec = pd.read_csv('../input/LS2009Electors.csv')
print(LS09Elec.shape)
LS09Elec.head()
# Reading 2014 Electors dataset
LS14Elec = pd.read_csv('../input/LS2014Electors.csv')
print(LS09Elec.shape)
LS14Elec.head()
LS09Elec.STATE.unique()
LS14Elec.STATE.unique()

LS14Elec['STATE']=LS14Elec['STATE'].replace(to_replace=['Odisha'],value='Orissa')
LS14Elec['STATE']=LS14Elec['STATE'].replace(to_replace=['Chhattisgarh'],value='Chattisgarh')

LS09Elec = LS09Elec.groupby('STATE').mean()
LS09 = LS09Elec[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y09=[2009 for i in range(35)]
S09=list(LS09['POLL PERCENTAGE'].keys())
P09=list(LS09['POLL PERCENTAGE'].values())


LS14Elec = LS14Elec.groupby('STATE').mean()
LS14 = LS14Elec[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y14=[2014 for i in range(35)]
S14=list(LS14['POLL PERCENTAGE'].keys())
P14=list(LS14['POLL PERCENTAGE'].values())
Data = {'YEAR':Y09+Y14,'STATE':S09+S14,'Poll_Percentage':P09+P14}
DF = pd.DataFrame(data=Data)
ax = plt.subplots(figsize=(6, 20))
sns.barplot(x=DF.Poll_Percentage,y=DF.STATE,hue=DF.YEAR)
plt.title('Poll Percentage of States 2009 and 2014')

