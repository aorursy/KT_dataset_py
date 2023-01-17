#imports



%matplotlib inline



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
#read csv file

df = pd.read_csv("../input/RA_cleaned.csv", delimiter=',', quotechar='"', header=0, index_col="ra_review_id", parse_dates=['review_published'])



#convert published date to datetime

df.review_published = pd.to_datetime(df.review_published)
print(df.groupby("release_type").rating.mean())
release_type_score_by_year = df.groupby(["release_year","release_type"]).rating.mean().unstack(level=-1)

yearly_average_score = df.groupby("release_year").rating.mean()



plt.figure(figsize=(9,4))

_ = plt.plot(release_type_score_by_year.album)

_ = plt.plot(release_type_score_by_year.single)

_ = plt.plot(yearly_average_score,linestyle="--",c='grey')



plt.xticks(rotation=60)

plt.xlabel("Year")

plt.ylabel("Average Score")

plt.legend(("Album","Single","All Releases"))

plt.title("Average Scores of Albums and Singles",size=15)

plt.show()
release_type_count_by_year = df.groupby(["release_year","release_type"]).artist.count().unstack(level=-1)



plt.figure(figsize=(9,4))

_ = plt.plot(release_type_count_by_year.album)

_ = plt.plot(release_type_count_by_year.single)





plt.xticks(rotation=60)

plt.xlabel("Year")

plt.ylabel("# of Releases")

plt.legend(("Album","Single"))

plt.title("Count of Albums and Singles Per Year", size=15)

plt.show()
#the japanese version of the site creates a separate review entry for some releases. This line removes those

df_cleaned = df.sort_values(['artist','release_title','num_comments'],ascending=True).drop_duplicates(["artist",'release_title'],keep='last')



#group by artists and count how many reviews each artist has. Sort descending. Then lop off the first two, as those are Various Artists or unknown

artist_count = df_cleaned.groupby("artist").rating.count().sort_values(ascending=False).iloc[2:]



#one more "Unknown Artist" placeholder artist to drop

artist_count.drop('Unknown Artist',inplace=True)





#create list of artists with over 10 releases 

artists_over_ten_releases = list(artist_count[artist_count > 10].index)



#compile average ratings for those artists

artist_average = df[df['artist'].isin(artists_over_ten_releases)].groupby("artist").rating.mean()



#merge them into the same dataframe

artist_average = artist_average.to_frame().reset_index()

artist_count = artist_count.to_frame().reset_index()

top_artists_releases_ratings = pd.merge(artist_average,artist_count,how="inner",on="artist").set_index("artist")

top_artists_releases_ratings.columns = ['Avg Rating',"Releases"]



artists_list = list(top_artists_releases_ratings.index)





sns.set_style("darkgrid")

plt.figure(figsize=(20,15))

plt.scatter(top_artists_releases_ratings["Releases"],top_artists_releases_ratings["Avg Rating"])



for i,txt in enumerate(artists_list):

            plt.annotate(str(artists_list[i]),

                     (top_artists_releases_ratings["Releases"].iloc[i] + 0.1,

                      top_artists_releases_ratings["Avg Rating"].iloc[i]),

                     rotation=7,size=10

                    )



plt.xlabel("Total Releases")

plt.ylabel("Average Score")

plt.title("Average Score of Artists with 10 or more Releases",size=20)

plt.show()

def ArtistRatingStats(artists_to_search,show_releases=True):

    

    #chop out relevant artists

    artist_results = df[df.artist.isin(artists_to_search)]

    

    #elimiate duplicates

    artist_results = artist_results.sort_values(['artist','release_title','num_comments'],ascending=True).drop_duplicates(["artist",'release_title'],keep='last')

        

    #create datetime array for release dates

    dates = pd.to_datetime(artist_results.release_month + " " +artist_results.release_year.apply(str)).values

    

    #create list of release artist and release title

    releases = (artist_results.release_title).values

    

    

    #get plotting data for each artist in list and plot

    for artist in artists_to_search:

        

        artist_rows = artist_results[artist_results.artist == artist]

        

        artist_dates = pd.to_datetime(artist_rows.release_month + " " +artist_rows.release_year.apply(str)).values

    

        plt.scatter(artist_dates,artist_rows.rating,s=(artist_rows.num_comments*10**2.6 + 300),alpha=.5)

    

    #create annotations

    if show_releases:

        for i,txt in enumerate(releases):

            plt.annotate(str(releases[i]),

                     (dates[i] ,artist_results.rating.iloc[i] + 0.01),

                     rotation=7,size=10

                    )

    

    #format plot and add labels

    plt.xlabel("Year", size=30)

    plt.ylabel("Release Score",size=30)

    plt.xticks(rotation=60,size=30)

    plt.yticks(size=30)

    

    title = ""

    for artist in artists_to_search:

        title += artist + " Vs. "

    title = title[:-4]

    plt.legend(artists_to_search,markerscale=.2,

               frameon=True,

               fancybox=True,

               shadow=True,

               facecolor="w",

               labelspacing=1,

               fontsize=14)

    plt.title(title,size=40)

    plt.show()

    

#figure controls

sns.set_style("darkgrid")

plt.figure(figsize=(35,15))



#function call

ArtistRatingStats(["Radio Slave","Andy Stott"],show_releases=True)
def LabelRatingStats(labels_to_search,show_releases=True):

    

    #chop out relevant labels

    label_results = df[df.label.isin(labels_to_search)]

    

    #elimiate duplicates

    label_results = label_results.sort_values(['artist','release_title','num_comments'],ascending=True).drop_duplicates(["artist",'release_title'],keep='last')

         

    #create datetime array for release dates

    dates = pd.to_datetime(label_results.release_month + " " +label_results.release_year.apply(str)).values

    

    #create list of release artist and release title

    releases = (label_results.artist + " - " + label_results.release_title).values

    

    

    #get plotting data for each label in list and plot

    for label in labels_to_search:

        

        label_rows = label_results[label_results.label == label]

        

        label_dates = pd.to_datetime(label_rows.release_month + " " +label_rows.release_year.apply(str)).values

    

        plt.scatter(label_dates,label_rows.rating,s=(label_rows.num_comments*10**2.6 + 300),alpha=.5)

    

    #create annotations

    if show_releases:

        for i,txt in enumerate(releases):

            plt.annotate(str(releases[i]),

                     (dates[i] ,label_results.rating.iloc[i] + 0.01),

                     rotation=7,size=10

                    )

    

    #format plot and add labels

    plt.xlabel("Year", size=30)

    plt.ylabel("Release Score",size=30)

    plt.xticks(rotation=60,size=30)

    plt.yticks(size=30)

    

    title = ""

    for label in labels_to_search:

        title += label + " Vs. "

    title = title[:-4]

    plt.legend(labels_to_search,markerscale=.2,

               frameon=True,

               fancybox=True,

               shadow=True,

               facecolor="w",

               labelspacing=1,

               fontsize=14)

    plt.title(title,size=40)

    plt.show()

    



#figure controls



sns.set_style("darkgrid")

plt.figure(figsize=(35,15))





#function call

LabelRatingStats(['Ghostly International',"XL Recordings"],show_releases=False)

#figure controls

sns.set_style("darkgrid")

plt.figure(figsize=(35,15))



#function call

LabelRatingStats(["The Bunker New York","Whities","Let's Play House"],show_releases=True)
df_styles = df[df['style'].notnull()].copy()



lower_case_electronic = df_styles['style'].replace("Electronic",'electronic')

df_styles['style'] = lower_case_electronic

def GenreRatingCompare(genres):

    

    #initialize lists and dataframe

    genre_rating_dataframes = []

    combine_ratings = pd.DataFrame()



    #create lists of average ratings per year on df rows that contain style in i element of list

    for i in range(0,len(genres)):

        genre_rating_dataframes.append(df_styles[df_styles['style'].str.contains(genres[i])].groupby('release_year').rating.mean())



    #add these rows back into a single dataframe

    for i in range(0,len(genres)):

        combine_ratings = pd.concat([combine_ratings,genre_rating_dataframes[i]],axis=1)



    #columns in the new dataframe will match the passed in list of genres

    combine_ratings.columns = genres



    #format the figure

    plt.figure(figsize=(20,10))

    sns.set_style("darkgrid")

    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.ylabel("Average score of release containng genre",fontsize=20)

    plt.title("Average Score of Sub-Genre per Year",fontsize=25)



    #plot combined dataframe

    _ = plt.plot(combine_ratings.dropna())



    #format legend

    plt.legend(genres,prop={'size': 15})

    
GenreRatingCompare(["Pop","Techno","House",'Experimental','Electro',"Tech House","Ambient"])
#simple group by year and aggrigate with standard deviation of the rating

std = df.groupby('release_year').rating.std()



#format and show graph

plt.figure(figsize=(9,6))

plt.xlabel("Year",fontsize=15)

plt.xticks(rotation=30)

plt.ylabel("Standard Deviation of Year's Releases",fontsize=15)

plt.title("SD of Releases over Year",fontsize=15)

_ = plt.plot(std)
def GenrePrecentCompare(genres):



  #initialize lists and dataframes

  genre_count_dataframes = []

  columns = []

  percents = pd.DataFrame()

  combine_counts = pd.DataFrame()



  #loop through provided styles, chop out release *containing* that style, group by year and count

  for i in range(0,len(genres)):

      genre_count_dataframes.append(df_styles[df_styles['style'].str.contains(genres[i])].groupby('release_year').release_type.count())



  #also track how many releases total per year

  all_count_by_year = df_styles.groupby("release_year").release_type.count()





  #create single dataframe from genre count series'

  for i in range(0,len(genres)):

      combine_counts = pd.concat([combine_counts,genre_count_dataframes[i]],axis=1)

      

      #also add entry to list for column names

      columns.append(genres[i])



  #add the all-releases column to dataframe

  combine_counts = pd.concat([combine_counts,all_count_by_year],axis=1)



  #add "all" column label to list of labels

  columns.append("all")



  #set dataframe labels to list of labels

  combine_counts.columns = columns



  #populate new dataframe -percents- with percentage values of releases containing genres from combine_counts dataframe

  for i in range(0,len(genres)):

      percents[genres[i]] = combine_counts[genres[i]] / combine_counts['all'] * 100



  #if no releases with that genre existed that year, replace with 0

  percents = percents.fillna(0)



  #format figure

  plt.figure(figsize=(20,10))

  sns.set_style("darkgrid")



  plt.legend(genres,

                 frameon=True,

                 fancybox=True,

                 shadow=True,

                 facecolor="w",

                 labelspacing=1,

                 fontsize=14)

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.ylabel("% Of Releases Containing Genre",fontsize=20)



  #plot dataframe

  _ = plt.plot(percents)



  #format legend

  plt.legend(genres,prop={'size': 15})



  plt.show()



GenrePrecentCompare(["Pop","Techno","House",'Experimental','Electro',"Tech House","Breakbeat"])
GenrePrecentCompare(["Break",'Minimal',"Deep House","Dubstep","Bass","Ambient","Grime"])
GenrePrecentCompare(["Grime"])
GenrePrecentCompare(["Garage"])