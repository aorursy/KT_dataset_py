# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df1 = pd.read_csv("../input/anime.csv")

df2 = pd.read_csv("../input/rating.csv")

# Any results you write to the current directory are saved as output.
#df1["genre"].value_counts()
#df1[df1["rating"]>1].groupby(["genre"])["rating"].mean().sort_values()
#df1[df1["anime_id"]==790]
#df1.sort_values("members",ascending=False)
genre_set=set()

for each in set(df1["genre"]):

    if type(each)==str:

        this_genres=each.split(",")

        for e in this_genres:

            genre_set.add(e.strip(" "))

#print(genre_set)



for gen in genre_set:

    print(gen + ": " + str(df1[df1["genre"].str.contains(gen)==True]["members"].mean()))
gen_freq = {}

for genre in genre_set:

    count = 0

    for this_genres in df1.genre:

        if type(this_genres) == str:

            if genre in this_genres.split(", "):

                count += 1

    gen_freq[genre] = count

    

sorted_gen = sorted(gen_freq, key=gen_freq.get, reverse=True)

sorted_gen_freq = {}

for w in sorted_gen:

    sorted_gen_freq[w]=gen_freq[w]

#print(sorted_gen_freq)
gen_rating = {}

for gen in genre_set:

    gen_rating[gen] = (df1[df1["genre"].str.contains(gen)==True]["rating"].mean())

    #print(gen + ": " + str(df1[df1["genre"].str.contains(gen)==True]["members"].mean()))

#print(gen_rating)

sorted_gen_rating = {}

for w in sorted_gen:

    sorted_gen_rating[w]=gen_rating[w]

#print(sorted_gen_rating)
gen_members = {}

for gen in genre_set:

    gen_members[gen] = (df1[df1["genre"].str.contains(gen)==True]["members"].mean())

    #print(gen + ": " + str(df1[df1["genre"].str.contains(gen)==True]["members"].mean()))

#print(gen_rating)

sorted_gen_members = {}

for w in sorted_gen:

    sorted_gen_members[w]=gen_members[w]

#print(sorted_gen_members)
d = {'_genre': sorted_gen, 'anime_number': list(sorted_gen_freq.values()),"rating_avg": list(sorted_gen_rating.values()), "members_avg":list(sorted_gen_members.values())}

dd = pd.DataFrame(data=d)

print(dd)

#print(dd)

#dd.plot.area()

#dd.plot.barh()

#dd.plot.density()

#dd.plot.hist()

#dd.plot.line()

#dd.plot.scatter(x="genre",y="anime_in_this_genre")

#dd.plot.bar()

#dd.plot.box()

#dd.plot.hexbin(x="genre",y="anime_in_this_genre")

#dd.plot.kde()

#dd.plot.pie(x="genre",y="anime_in_this_genre")
df1[df1["rating"]>0].groupby(["type"])["rating"].mean().sort_values()
type_rating = df1[df1["rating"]>0].groupby(["type"])["rating"].mean().sort_values() 

type_rating.sort_values().plot.bar(x='type')

dd.sort_values("members_avg",ascending=False).head(n=10).plot.bar(x='_genre', y='members_avg')

dd.sort_values("rating_avg",ascending=False).head(n=10).plot.bar(x='_genre', y='rating_avg')

dd.head(n=10).plot.bar(x='_genre', y='anime_number')

dd.plot.scatter("members_avg","rating_avg",alpha=0.5)
df2.rating.plot.hist()