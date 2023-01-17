""" HEY GUYS! This is my first kaggle project and is made for complete beginners. I worked on this by myself after two months of teaching myself python and jupyter.

So I had lot of frustrations especially with coding erros but my love for fifa helped me out:). Kindly let me know what you think and suggestions are the best! """



# I have added the citations at the end of the code.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv("../input/fifa19/Fifa_19.csv")

#To find out the mean values of our data set 

df.describe()
#To find the first five top values

df.head()
#To find the unique body types

df['Body Type'].unique()
df.columns.unique()
#To remove the unwanted columns from our dataset

df = df.drop(columns = ['ID','Special','International Reputation','Body Type','Real Face','Joined','Loaned From','Contract Valid Until','Weight','Photo','Flag','Club Logo','Release Clause'])

        
#To check the new data set

df.head()
#Lets try replacing our index with player name

df = df.set_index("Name")

df.head(2)
#To find out if there are any null values in our dataset

df.isnull().sum()
#To fill out the null values with the mean

df=df.fillna(df.mean())

df.isnull().sum()
def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)

# applying the function to the wage column



df['Value'] = df['Value'].apply(lambda x: extract_value_from(x))

df['Wage'] = df['Wage'].apply(lambda x: extract_value_from(x))



df['Wage'].head()



#Some data trimming for height column

df[df.Height.isnull()].Height

df["Height"].fillna("0'0", inplace = True)



#Now to convert the values from inches to cm

newlist = []

for x in df.Height:

 a = str(x)

 b = a.split("'")

 c = b[0]

 d = b[1]

 e = round(float(int(c)*12+int(d))*2.54,2)

 newlist.append(e)



#You can convert this into an array if it shows Type-error    

New = np.array(newlist)    

df["Heightcm"] = pd.DataFrame(New,index = df.index)



#Lets remove our last height column

df = df.drop(columns = ['Height'])



#Finally lets replace the null values by mean height

df["Heightcm"].replace(to_replace = 0, value = df["Heightcm"].mean(), inplace = True)
#Lets see the distribution of height using histogram

import matplotlib.pyplot as plt



x = df.Heightcm

plt.hist(x, bins = 100)

plt.xlabel('Height')

plt.ylabel('Frequency')

plt.title('Distribution of Height')

plt.show()
#To find the maxmimum count of a particalur column

df["Wage"].value_counts()
#Lets find the wage distribution but lets use seaborn this time



plt.figure(figsize=(16, 6))

sns.distplot(df.Wage);

#If you feel the graphs need a better view you can always switch to matplot. I am using seaborn here just for demo.
#To find the player for each highest attribute(Overall, Potential, Wage, Age, Height



print(f"{df.index[df['Overall'].argmax()]} has an overall of {df['Overall'].max()}")

print(f"{df.index[df['Overall'].argmin()]} has an overall of {df['Overall'].min()}")

print(f"{df.index[df['Potential'].argmax()]} has potential of {df['Potential'].max()}")

print(f"{df.index[df['Potential'].argmin()]} has potential of {df['Potential'].min()}")

print(f"{df.index[df['Age'].argmax()]} has age of {df['Age'].max()}")

print(f"{df.index[df['Age'].argmin()]} has age of {df['Age'].min()}")

print(f"{df.index[df['Heightcm'].argmax()]} has an height of {df['Heightcm'].max()} cm")

print(f"{df.index[df['Heightcm'].argmin()]} has an height of {df['Heightcm'].min()} cm")

print(f"{df.index[df['Value'].argmax()]} has a value of {df['Value'].max()} EUR")

print(f"{df.index[df['Value'].argmin()]} has a value of {df['Value'].min()} EUR")

print(f"{df.index[df['Wage'].argmax()]} has a wage of {df['Wage'].max()} EUR")

print(f"{df.index[df['Wage'].argmin()]} has a wage of {df['Wage'].min()} EUR")
df["Skill Moves"].unique()
df["Skill Moves"] = df["Skill Moves"].astype(int)
#Now lets plot some correlation plots between our parameters

sns.lmplot(x='Overall', y='Wage', data=df,

           fit_reg=True,

           hue='Skill Moves').set(Title='Overall vs Wage wrt Skill moves')

sns.lmplot(x='Overall', y='Wage', data=df,

           fit_reg=True,

           hue='Preferred Foot').set(Title='Overall vs Wage wrt Preferred Foot')
#Lets try to do some box plots

cols = ['Finishing','HeadingAccuracy','ShortPassing','Dribbling','LongPassing','Acceleration','ShotPower','Stamina','LongShots','Marking','StandingTackle','GKDiving','GKPositioning']

plt.figure(figsize=(20, 6))

sns.boxplot(data=df[cols]).set(Title = "Comparion of player attributes")
#To find out the mean of our newly updated columns

print(f"The average value of players is {df['Value'].mean()} EUR")

print(f"The average wage of players is {df['Wage'].mean()} EUR")

print(f"The average height of players is {df['Heightcm'].mean()} cm")
#Lets search our players based on nationality and find the top five players

df[df['Nationality']=='Brazil'].Overall.sort_values(ascending = False).head(5)
#Now lets make a distribution showing the wages of players in different countries



cols = ('Spain','England','Italy','Brazil','Germany','Japan','Sweden','Ivory Coast','Australia')

cols_loc = df.loc[df['Nationality'].isin(cols) & df['Wage']]

plt.figure(figsize =(10, 20))

sns.swarmplot(x = cols_loc['Nationality'], y = cols_loc['Wage']).set(Title = "Comparison of wages between men of different nationality")

plt.xlabel('Countries')

plt.ylabel('Wage')



#Shows that players from England are payed well above average
#Lets filter our players who are left footed and has an overall greater than 85 and who is from either Argentina or Uruguay

#data[(data.Origin=='Asia')|(data.Origin=='Europe')]

df[(df['Preferred Foot']=='Left')&(df['Overall'] > 85)&((df.Nationality =='Argentina')|(df.Nationality=='Uruguay'))]



#Tip : How you place the brackets ("Is very important")
#Lets see the ratio of players of different work rates

types = ['Medium/ Medium', 'High/ Low', 'High/ Medium', 'High/ High',

       'Medium/ High', 'Medium/ Low', 'Low/ High', 'Low/ Medium',

       'Low/ Low']

works = df["Work Rate"].value_counts()

plt.title('Count of different types of workrates')

plt.pie(works, labels = types, radius = 1,autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)

plt.figure(figsize =(20, 20))
#Lets see which jersey number shows the most overall.



cols = (1,4,6,10,9,7,20,30,90,99)

cols_loc = df.loc[df["Jersey Number"].isin(cols) & df['Overall']]

plt.figure(figsize =(15, 20))

sns.violinplot(x = cols_loc['Jersey Number'], y = cols_loc['Overall'],cut=5).set(Title = "Comparison of jersey number vs overall")

plt.xlabel('Jersey Number')

plt.ylabel('Overall')



#Seems like the jersey number 10 gets paid the most
#Lets do a swarm plot of weak foot vs  overall

df["Weak Foot"] = df["Weak Foot"].astype(int)

plt.figure(figsize=(20, 6))

sns.violinplot(x= "Weak Foot", y = "Overall", data = df).set(Title = "Comparison of overall vs weak foot")

#Seems like the overall increases as the players improve their weak foot
#Since this is Fifa 19 you may wanna increase the age of your players by 1 or 2(For the geeks who know fifa 19 is released in 2018)

#But this is clearly data manipulation and just for learning purpose. You can revert the values back if needed.

df['Age']=df['Age'].apply(lambda x:x+2)

df['Age']=df['Age'].apply(lambda x:x-2)
#lets try to find the mean overall of a club

df.groupby('Club').Overall.mean().sort_values(ascending = False).head(10)
#Lets count the total number of players in each position

plt.figure(figsize=(20, 6))

sns.countplot(x='Position', data=df).set(Title = "Count of players in each position")
#Lets find the top 5 richest clubs based on its players value

df.groupby("Club").Value.mean().sort_values(ascending = False).head(5)
#Lets plot the above richest clubs based on its player value

cols = ('FC Barcelona', 'Juventus', 'Paris Saint-Germain',

       'Manchester United', 'Manchester City', 'Chelsea', 'Real Madrid','FC Bayern München','Arsenal', 'Milan')

cols_loc = df.loc[df["Club"].isin(cols) & df['Value']]

plt.figure(figsize =(20, 20))

sns.violinplot(x = cols_loc['Club'], y = cols_loc['Value']).set(Title = "Richest clubs in fifa 19")

plt.xlabel('Club')

plt.ylabel('Value')
#Compare the clubs in terms of most number of young potential players

a = df.loc[(df["Potential"]>85)&(df["Age"]<21)]

a["Count"] = [1]*66

a.groupby("Club").Count.sum().sort_values(ascending = False)

#Seems this warning is common with this software version at the moment. Let me know if we can improve the code in some way.



#As you can see Borussia Dortmund has or creates the most number of promising young players
#Lets make a correlation graph between the different parameters

cols = ['Age','Overall','Potential','Wage','Value','Heightcm']

stats = df[cols]



corr = stats.corr()

sns.heatmap(corr).set(Title ="Correlation matrix" )
#Lets do a pairplot between the same parameters

cols = ['Overall','Potential','Wage','Value']

sns.pairplot(df[cols]);
#Relationship of overall vs wage of a player with different weak foots

sns.catplot(x='Overall', 

                   y='Wage', 

                   data=df,

                   hue = 'Preferred Foot',

                   col='Preferred Foot', 

                   kind='swarm') 
#To find a specific player attributes. Lets try it for Messi.

df.loc['L. Messi']
#Now lets make a spider plot to show the stats of a particular player

#Before that lets revert the index back to its orignal position



df.reset_index(inplace = True) 

#Now lets continue..



%matplotlib inline

import pandas as pd

import seaborn as sns

import numpy as np



labels=np.array(['Finishing', 'Dribbling', 'Acceleration', 'BallControl', 'HeadingAccuracy', 'ShotPower'])

stats=df.loc[0,labels].values

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))



fig=plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.2)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title([df.loc[0,'Name']])

ax.grid(True)
"""cite:

https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way

https://seaborn.pydata.org/tutorial/distributions.html

https://elitedatascience.com/python-seaborn-tutorial """