import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline 
fifa=pd.read_csv("../input/fifa19/data.csv")
fifa.head()
fifa.columns
bins=[0,10,20,30,40,50,60,70,80,90,100]

plt.hist(fifa['Overall'],bins)

#plt.xticks(bins)

plt.xlabel("Skill Level")

plt.ylabel("No of Players")

plt.title("Distribution of player skills in fifa")

plt.show()
bins=[80,90,100] 

plt.hist(fifa['Overall'],bins)



plt.xlabel("Skill Level")

plt.ylabel("No of Players")

plt.title("Distribution of player skills in fifa")

plt.show()
bins=[80,90,100] #for zooming data in particular sections

plt.hist(fifa['Overall'],bins,log=True,edgecolor="white")

#plt.xticks(bins)

plt.xlabel("Skill Level")

plt.ylabel("No of Players")

plt.title("Distribution of player skills in fifa")

plt.show()
bins=[80,90,100] 

plt.hist(fifa['Overall'],bins,log=True,edgecolor="white")

#plt.xticks(bins)

plt.xlabel("Skill Level")

plt.ylabel("No of Players")

plt.title("Distribution of player skills in fifa")

plt.xticks(bins)

plt.show()
bins=[40,50,60,70,80,90,100]

plt.figure(figsize=(8,5))

plt.hist(fifa['Overall'],color="lightblue")

#plt.xticks(bins)

plt.xlabel("Skill Level")

plt.ylabel("No of Players")

plt.title("Distribution of player skills in fifa")

plt.xticks(bins)



plt.show()

#plt.hist(fifa['Preferred Foot'])

#plt.savefig("preferredfoot.jpg")



#plt.show()

#total number of club present and top 5 clubs with highest number of player

print("total number of clubs:{0}".format(fifa['Club'].nunique())) #nunique will give number of unique values

print(fifa['Club'].value_counts().head(5))
fifa['Nationality'].nunique() #number of countries
print(fifa['Nationality'].value_counts().head(5)) #gives top 5 countries
fifa['Preferred Foot'].value_counts()
#plt.hist(fifa['Preferred Foot'])

#plt.show()
#fifa['Potential'].max()

fifa[fifa['Potential']==fifa['Potential'].max()]['Name']
#fifa['Overall'].max()

print(fifa[fifa['Overall']==fifa['Overall'].max()]['Name'].head(1).to_string(index=False))
print("Maximum Potential: "+str(fifa.loc[fifa['Potential'].idxmax()][2]))

print("Maximum Overall Performance: "+str(fifa.loc[fifa['Overall'].idxmax()][2]))
ax=sns.countplot(x='Skill Moves',data=fifa) #for bar plot in seaborn
plt.figure(figsize=(7,8))

ax=sns.countplot(x='Skill Moves',data=fifa,palette='pastel')

ax.set_title(label='Count of players on basis of their skill moves',fontsize=20)

ax.set_xlabel(xlabel="Number of Skill Moves",fontsize=16)

ax.set_ylabel(ylabel="Count",fontsize=16)

plt.show()
plt.figure(figsize=(7,8))

sns.set(style="dark",palette="colorblind",color_codes=True)

x=fifa.Age

ax=sns.distplot(x,bins=58,kde=True,color='Orange') #kde will show line

ax.set_title(label='Histogram of player age as their age',fontsize=20)

ax.set_xlabel(xlabel="Player Age",fontsize=16)

ax.set_ylabel(ylabel="Number of Players",fontsize=16)

plt.show()
some_clubs=['Juventus','Real Madrid']

fifa_club=fifa.loc[fifa['Club'].isin(some_clubs)&fifa['Age']]

fig,ax=plt.subplots()

fig.set_size_inches(15,10)

ax=sns.violinplot(x="Club",y="Age",data=fifa_club)

ax.set_title(label="Distibution of Players age in some clubs",fontsize=20)
plt.figure(figsize=(12,8))

ax=sns.countplot(x='Position',data=fifa)

ax.set_title(label='Comparison of postions and players',fontsize=20)

ax.set_xlabel(xlabel="Different position in football",fontsize=16)

ax.set_ylabel(ylabel="Count of players",fontsize=16)

plt.show()