# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading data in dataframes

data1=pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

data2=pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')

data1.info()

data2.info()
# Making a new dataframe genre containing only artists and genre. Cleaning the dataframe.



genre=data1[["genres","artists"]]

genre=genre[genre["genres"]!="[]"]

genre["genres"]=genre["genres"].str.replace("'", "")

genre["genres"]=genre["genres"].str.replace("[", "")

genre["genres"]=genre["genres"].str.replace("]", "")
# Exploring Genre dataframe

genre.head(50)
#Exploring the most popular genre



genre_dict={}



genre_df_dict=genre["genres"].str.split(",")

for index, genre_list in genre_df_dict.iteritems():

    for genre_name in genre_list:

        if genre_name in genre_dict:

            genre_dict[genre_name]+=1

        else:

            genre_dict[genre_name]=1

            

genre_dict_sorted=sorted(genre_dict.items(), key= lambda x:x[1], reverse=True) 

genre_dict_sorted
pattern_classical='orchestra[\s\S]*|\b?[\s\S]*classical[\s\S]*\b?|[\s\S]*piano[\s\S]+|opera'

pattern_movies=r'\bbroadway[\s\S]*|\b[\s\S]*movie[\s\S]+|\b[\s\S]*show[\s\S]+|\b[\s\S]*hollywood'

pattern_kpop=r'[\s\S]+k[-]?pop[\s\S]+'

pattern_rock=r'[\s\S]+ rock[\s\S]*|[\s\S]+metal[\s\S]*|[\s\S]+punk[\s\S]*'

pattern_pop=r'[\s\S]+pop[\s\S]*'

pattern_rap=r'[\s\S]* rap[\s\S]*'

pattern_jazz=r'[\s\S]*jazz[\s\S]*|bossa nova'

pattern_reggae=r'\b[\s\S]*reggae[\s\S]*'

pattern_r=r'[\s\S]*christian[\s\S]*|[\s\S]*gospel[\s\S]*'

pattern_folk=r"[\s\S]*folk[\s\S]*|\brebetiko|hawaiian, jawaiian|ukulele|duranguense|grupera|ranchera|[\s\S]*regional mexican|norteno|latin"

pattern_country=r'[\s\S]*country[\s\S]*|\btejano\b'

pattern_rnb=r'[\s\S]*blues[\s\S]*|[\s\S]*r&b[\s\S]*'

pattern_funk=r'[\s\S]*funk[\s\S]*|[\s\S]*disco[\s\S]*|[\s\S]*hip hop[\s\S]*'

pattern_edm=r'electronica|[\s\S]*edm[\s\S]*|electro house|electronic trap'







genre["genres_main"]=genre["genres"].str.replace(pattern_classical,"classical")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_movies,"movies_and_broadway")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_r,"religiuos")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_kpop,"kPop")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_reggae,"reggaeton")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_rap,"rap")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_rock,"rock")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_pop,"pop")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_jazz,"jazz")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_country,"country music")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_folk,"folk")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_rnb,"rhytm_and_blues")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_funk,"funk")

genre["genres_main"]=genre["genres_main"].str.replace(pattern_edm,"edm")





# Finding main genres to select for analysis



pd.set_option("display.max_rows",None)

genre["genres_main"].value_counts()
genre_dict_a={}



genre_df_dict_a=genre["genres_main"].str.split(",")

for index, genre_list_a in genre_df_dict_a.iteritems():

    for genre_name_a in genre_list_a:

        if genre_name_a in genre_dict_a:

            genre_dict_a[genre_name_a]+=1

        else:

            genre_dict_a[genre_name_a]=1

genre_dict_sorted_a=sorted(genre_dict_a.items(), key= lambda x:x[1], reverse=True)

genre_dict_sorted_a
# Creating a dictionary which categorises an artist's work into a main genre of music



artist_dictionary={}

for index, row in genre.iterrows():

    artist_name=row["artists"]

    artist_genre=row["genres_main"]

    artist_dictionary[artist_name]=artist_genre
#Adding a genre column in data2 by using information in dictionary created above



data2["artists"]=data2["artists"].str.replace("\[","")

data2["artists"]=data2["artists"].str.replace("\]","")

data2["artists"]=data2["artists"].str.replace(", ",",")

data2["artists"]=data2["artists"].str.split(",")







def find_genre(column):

    music_style=[]

    for artist in column:

        artist=artist.strip("'")

        if artist in artist_dictionary:

            #print(artist)

            music_style.append(artist_dictionary[artist])

    return music_style

            

data2["genre"]=data2["artists"].apply(find_genre)

data2.info()
# Looking at information in genre column

data2['genre'].describe()
data2["genre"]=data2["genre"].astype(str)

data2["genre"]=data2["genre"].str.replace("]","")

data2["genre"]=data2["genre"].str.replace("[","")
# Selecting only those rows whose genre information is available.

data2=data2[data2["genre"]!=""]

data2["genre"]=data2["genre"].str.strip("'")

data2["genre"]=data2["genre"].str.replace("', '", ",")

data2["genre"]=data2["genre"].str.replace(r"classical,{1,}[\s\S]*","classical")

data2["genre"]=data2["genre"].str.replace(r"movies_and_broadway,{1,}[\s\S]*","movies_and_broadway")

data2["genre"]=data2["genre"].str.replace(r"[\s\S]*bollywood[\s\S]*|[\s\S]*filmi[\s\S]*","bollywood")

data2["genre"]=data2["genre"].str.replace(r"\brap,+rap\b","rap")

data2["genre"]=data2["genre"].str.replace(r"\bfolk,+folk\b|folkfolk","folk")

data2["genre"]=data2["genre"].str.replace(r"jazz,+jazz\b","jazz")

data2["genre"]=data2["genre"].str.replace(r"\bpop,+pop\b","pop")

data2["genre"]=data2["genre"].str.replace(r"rock,rock","rock")

data2["genre"]=data2["genre"].str.replace(r"[\s\S]*folk, ?mariachi, folk","folk")

data2["genre"]=data2["genre"].str.replace(r"reggaeton,reggaeton","reggaeton")

data2["genre"]=data2["genre"].str.replace(r"country music,country music","country music")

data2["genre"]=data2["genre"].str.replace(r"rap,+rap","rap")
data2["genre"].value_counts()


genre_list_interest=["rock","classical","pop","jazz","rap","folk","country music","funk","reggaeton","movies_and_broadway",

                     "religiuos","rhytm_and_blues","bollywood","kPop"]

data2_new=data2[data2["genre"].isin(genre_list_interest)]
# Cleaning of 'artists 'column



data2_new["artists"]=data2_new["artists"].astype(str)

data2_new["artists"]=data2_new["artists"].str.strip('[]"')

data2_new["artists"]=data2_new["artists"].str.replace('\'','').str.replace('"',"")
data2_new["duration_ms"]=data2_new["duration_ms"]/1000

data2_new["duration_ms"].value_counts(bins=30)
# Creating a list of time values which will replace the duration_ms column for each row

list_time=[x for x in range(60,1740,60)]

list_time_2=[x for x in range (1740,3840,240)] # Longer songs are less, so larger intervals created.

list_time+=list_time_2



#Replace values of column"duration_ms" using  list_time

def round_duration(value):

    for time in list_time:

        if value<=time:

            new_value=time

            break

    return new_value



data2_new["duration_ms"]=data2_new["duration_ms"].apply(round_duration) 
# Aggregating data2

grouped=data2_new.groupby(['artists','genre','year'], as_index=False)
group_mean=grouped["duration_ms","popularity"].mean()

group_count=grouped["name"].count()

group_mean

group_mean["genre"]=group_mean["genre"].astype(str)

group_mean["song_count"]=group_count["name"]

group_mean.head()
import plotly

import plotly.express as px

import plotly.io as pio
# Sorting table by year

group_mean=group_mean.sort_values('year')

missing_data=group_mean[group_mean['year']==1921]

present_list=missing_data['genre'].unique().tolist()



for genre in genre_list_interest:

    if genre not in present_list:

        dict_new={'artists': "blank", 'genre': genre, 'year': 1921,'duration_ms':100,'popularity':0, 'song_count':0}

        group_mean=group_mean.append(dict_new, ignore_index=True)

# Selecting color scheme from plotly color charts

print(px.colors.qualitative.Alphabet)
color_discrete_map={'classical':'#AA0DFE', 'pop':'#3283FE', 'movies_and_broadway':'#85660D', 'jazz':'#16FF32', 'rock':'#565656',

       'religiuos': '#1C8356', 'funk':'#782AB6', 'reggaeton':'#FA0087', 'country music':'#FE00FA',

       'rhytm_and_blues':'#1CBE4F', 'folk':'#C4451C','rap':'#DEA0FD','kPop':'#B00068'}

fig=px.scatter(data_frame=group_mean,x="popularity",y="song_count",color="genre",size="duration_ms",size_max=30,\

               color_discrete_map=color_discrete_map,\

               hover_name="artists",animation_frame="year",range_x=[0,100],range_y=[0,20],\

              title="SPOTIFY : MUSIC TRENDS FROM 1921 TO 2020",labels={"popularity":"popularity","song_count":"song_count"})

fig.update_layout(title={'x':0.5,'xanchor':'center','font':{'size':20}},

                  xaxis={'title': {'text': 'POPULARITY'}},

                  yaxis={'title': {'text': 'NUMBER OF SONGS'}},

                  legend={'font':{'size':18},'title':{'font':{'size':18}}})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800

pio.show(fig)
# Creating new dataframe by aggregation 



missing_data=data2_new[data2_new['year']==1921]

present_list=missing_data['genre'].unique().tolist()



for genre in genre_list_interest:

    if genre not in present_list:

        dict_new={'artists': "blank", 'genre': genre, 'year': 1921,'duration_ms':100,'popularity':0, 'song_count':0}

        data2_new=data2_new.append(dict_new, ignore_index=True)

data2_new=data2_new.sort_values('year')
group_mean.head()
grouped_2=group_mean.groupby(['genre','year'],as_index=False)

df_a=grouped_2['artists'].count()

df_b=grouped_2['song_count'].sum()

df_c=grouped_2['popularity'].mean()

df_a['song_count']=df_b['song_count']

df_a['popularity']=df_c['popularity']

df_a=df_a.sort_values('year')
years_df_a=[i for i in range(1922,2021)]

for genre in genre_list_interest:

    test_df=df_a[df_a["genre"]==genre]

    test_list= test_df['year'].unique().tolist()

    for year in years_df_a:

        if year not in test_list:

            dict_new_1={'artists': 0, 'genre': genre, 'year': year,'duration_ms':0,'popularity':0, 'song_count':0}

            df_a=df_a.append(dict_new_1, ignore_index=True)

            
color_discrete_map={'classical':'#AA0DFE', 'pop':'#3283FE', 'movies_and_broadway':'#85660D', 'jazz':'#16FF32', 'rock':'#565656',

       'religiuos': '#1C8356', 'funk':'#782AB6', 'reggaeton':'#FA0087', 'country music':'#FE00FA',

       'rhytm_and_blues':'#1CBE4F', 'folk':'#C4451C','rap':'#DEA0FD','kPop':'#B00068'}

fig=px.scatter(data_frame=df_a,x="popularity",y="song_count",color="genre",size="artists",size_max=60,\

               color_discrete_map=color_discrete_map,\

               hover_name="genre",animation_frame="year",range_x=[0,90],range_y=[0,2000],\

              title="SPOTIFY: TRENDS OF MUSIC FROM 1921 TO 2020",labels={"popularity":"popularity","song_count":"song_count"})

fig.update_layout(title={'x':0.5,'xanchor':'center','font':{'size':20}},

                  xaxis={'title': {'text': 'popularty of each genre'}},

                  yaxis={'title': {'text': 'number of songs in each genre'}},

                  legend={'font':{'size':18},'title':{'font':{'size':18}}})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900

pio.show(fig)