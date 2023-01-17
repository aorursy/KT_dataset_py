import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth',-1)





#Loading the dataset

top = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='latin-1')



#Summary of dataset

top.describe()



#Printing the column names

print(top.columns)



top.drop(['Unnamed: 0'], axis=1,inplace=True)

#dropping the first column



top.rename(columns={'Loudness..dB..': 'Loudness', 'Valence.': 'Valence',

                   'Length.': 'Length','Acousticness..':'Acousticness','Speechiness.':'Speechiness'}, inplace=True)



#Dropped column 1 and renamed the columns of convenience

top.columns



top.iloc[32,0]='Name_Unavailable'



#Checking for Missing values in data

top.isna().sum()



#Printing the number of tracks per artist

top['Artist.Name'].value_counts().nlargest(10)



#Printing the top 20 songs based on their popularity

top.groupby(['Energy','Track.Name','Genre','Beats.Per.Minute','Popularity']).Popularity.mean().nlargest(20)



#Plotting the graph to check which Genres are more Popular

aw=sns.barplot(data=top, x='Genre', y = 'Popularity')

#aw.set_xticklabels(rotation=30)

plt.xticks(rotation=90)

plt.figure(figsize=(50,30))



#Printing the above plot in different way with median of Popularity for better understanding

test=top.groupby('Genre').Popularity.median()

final=test.to_frame()

final.style.background_gradient(cmap='Greys')



#The above style shows - dark colored boxes are highly popular ones: dfw rap,electropop,escape room, trap music, pop house



#Printing the top 10 Popular genres by taking count of songs



top['Genre'].value_counts().nlargest(10)



#Checking corelation between variables

sns.heatmap(top.corr(),cmap="Purples")

#Loudiness and energy are highly correlated



#Plotting individual histogram plots to check distribution

columns=['Beats.Per.Minute',

       'Energy', 'Danceability', 'Loudness', 'Liveness', 'Valence',

       'Length', 'Acousticness', 'Speechiness', 'Popularity']

for i in columns:

    plt.figure()

    top[i].hist()

    plt.title(i)

    

#Displays top 10 danceable songs

top.sort_values(by="Danceability",ascending=False).head(10)



#Top 10 high energy songs

top.sort_values(by="Energy",ascending=False).head(10)



#Binning the BPM to multi level

top['BPM_Level']=pd.cut(top['Beats.Per.Minute'], bins=[80, 95,110,125, 140,155,170,190],

                  labels=['Level 1','Level 2','Level 3','Level 4','Level 5','Level 6','Level 7'])



#Displaying count of songs for Different BPM Levels

sns.countplot(data=top,x="BPM_Level")

#Printing the dominating Genres across BPM Levels Level 1 and 2



levels2=['Level 1','Level 2']



for k in levels2:

    print(k)

    print(top[top['BPM_Level']==k]['Genre'].value_counts())

    print("************************")

    

aw=sns.barplot(data=top, x='BPM_Level', y = 'Popularity')

#lets check with numbers



top.groupby('BPM_Level').Popularity.median()

#As per the count of Level 5,6 and 7 Beats per minute, there are few songs in this range but they are quite popular ones. these songs have more beats in the range of 155-190



#Plotting Valence to popularity

sns.jointplot(data=top, x='Valence',y='Popularity',kind='scatter',color='b')



#Plotting Liveliness to popularity

sns.jointplot(data=top, x='Liveness',y='Popularity',kind='scatter',color='g')



# Conclusions: 

# The top 50 have more songs which are like party songs with more value of Danceability field.

# The songs are medium to fast pace songs, more on the medium side. there are no or very less slow songs in top 50 as per the 

# beats per minute value.



# Songs from latin, dance pop, pop, canadian hip hop, edm Genres are very popular here.



# Most of the songs are studio recorded ones and very few are live recorded songs, this could be due to the noise/clarity 

# issues with live recorded songs.



# Most of the songs time ranges between 150-225 seconds, that would be 2.5 to 3.5 minutes. People have listened to songs 

# which are not too short or too long. Maybe because the short songs end too soon and long songs get boring..



# There is correlation between energy and loudiness of song -> may be loud songs sound more energetic.



# Both positive and negative songs have almost equally made to the top50 list. There are equal happy and sad songs.



# The distribution of Speechiness plot tells that most of songs have less of words and more Music.