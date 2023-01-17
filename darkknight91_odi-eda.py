import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

import os

import time

from geopy.geocoders import Nominatim

from tqdm import tqdm_notebook as tqdm

from matplotlib import pyplot as plt

from matplotlib.pyplot import pie

import pickle

import re



print(os.listdir("../input"))

INPUT_PATH = "../input/cricket-odi-dataset-from-cricinfo"

# DRIVE_PATH = "/content/drive/My Drive/ML_Training_Data"  # variable name is remnant of colab kernel

DRIVE_INPUT_PATH = "../input/glove-global-vectors-for-word-representation"  # variable name is remnant of colab kernel

print(os.listdir("../input/cricket-odi-dataset-from-cricinfo"))
# All methods in one place



def print_runtime(text, stime):

    delta = time.time() - stime

    print(text, int(delta//60), "minutes", ":", int(np.round(delta%60, 0)), 

          "seconds")

    

    

def fill_country(df, write_to_file=True, file_name="ODI_cricinfo_country.csv"):

    """

    Original dataset only had 'Ground' detail. This method was used to 

    fill the country where the match took place. Then the dataframe is saved as

    CSV for future usage.

    """

    city_country_map = None

    try:

        city_country_map = pickle.load(open(os.path.join(INPUT_PATH, "city_country_map"), "rb"))

    except:

        print("saved city country map not found")

        

    if city_country_map is None:

        print("Geocoding city to country...")

        city_country_map = {}

        cities = list(df["Ground"].unique()) 

        print("Fetching country for", len(cities), "cities")

        geolocator = Nominatim(user_agent="kaggle")

        c = 0

        for city in tqdm(cities):

            # print(c,city)

            city = re.sub(r"\(.*\)","",city).strip()

            location = geolocator.geocode(city, language='en')

            if location is not None:

                city_country_map[city] = (location.address.split(', ')[-1])

            else:

                print(c, city)

            c = c+1

            time.sleep(1)  # to avoid violating usage policy

        pickle.dump(city_country_map, open(os.path.join(INPUT_PATH, "city_country_map"), 'wb'))

    

    # now add these countries as column

    countries = [city_country_map[re.sub(r"\(.*\)","",city).strip()] 

                 for city in tqdm(df["Ground"])]

    print(len(df), len(countries))

    print(df["Ground"][:10], countries[:10])

    df["HostCountry"] = countries

    if write_to_file:

      print("Writing dataframe to CSV")

      df.to_csv(file_name)

      

    

def add_feature_value_fequencies(df_local, col1, col2):

    """

    This method adds the frequencies of each value of passed features of 

    given dataframe.

    

    return: dictionary --> with count of each value of passed features combined

    """

    team_series_temp = df_local.groupby(col1)[col1].aggregate("count")

    team1_dict = team_series_temp.to_dict()

    # print(team1_dict)

    team_series_temp = df_local.groupby(col2)[col2].aggregate("count")

    team2_dict = team_series_temp.to_dict()

    # print(team2_dict)

    return Counter(team1_dict)+Counter(team2_dict)

  



def generate_colors_random(no_of_colors):

    """

    This method generates colors with RGB compnents randomly

    

    return: list of tuples(R, G, B)

    """

    RGB_tuples = []

    for i in range(no_of_colors):

        temp = np.random.uniform(0, 1, 3)

        RGB_tuples.append((temp[0], temp[1], temp[2]))

    return RGB_tuples

  



def generate_colors_by_magnitude(values):

    """

    This method generates colors with RGB compnents proportional

    to numbers in the list passed as parameter

    

    return: list of tuples(R, G, B)

    """

    min_ = min(values)

    max_ = max(values)

    new_max = 1

    new_min = 0

    r = 0.8

    g = 0.7

    b = 0.0 # can be changed as per preference

    colors = []

    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

    for value in values:

        range_shifted_val = (((value - min_) * (new_max - new_min)) / 

                             (max_ - min_)) + new_min

        _r = r * range_shifted_val

        _g = g * (new_max - range_shifted_val)

        _b = _r * _g / 1.5

        colors.append((_r, _g, _b))

        

    return colors

  

    

def breakline_labels(labels):

    """

    replaces space in a text with new line.

    """

    short_labels = []

    for l in labels:

        short_labels.append(l.replace(" ","\n"))

    return short_labels



def shorten_labels(labels):

    """

    formats the labels to a more suitable size ideal for 

    displaying in plots.

    """

    short_labels = []

    for label in labels:

        final = ""

        if " " in label:

            words = label.split()

            for w in words:

                final = final + w[0]

        elif len(label) > 6:

             final = label[0:3]

        else:

            final = label

        short_labels.append(final)

    return short_labels

    



def swap_columns(row, col1, col2):

    """

    Function called for dataframe row. Passed row is sorted alphabetically on axis = 1

    i,e. column values are sorted.

    """

    

    if row[col1] > row[col2]:

        temp = row[col1]

        row[col1] = row[col2]

        row[col2] = temp

    return row
stime = time.time()



# IMPORTANT!

# This step is already done once. I have already saved the result as a new CSV 

# and it's accessible as new dataset "ODI_cricinfo_country_en.csv"



# df = pd.read_csv(os.path.join(INPUT_PATH, "ODI_cricinfo_country.csv"))

# fill_country(df, True, "ODI_cricinfo_country_en.csv")



df = pd.read_csv(os.path.join(INPUT_PATH, "ODI_cricinfo_country_en.csv"))

# pickle.dump(df, open("ODI_cricinfo_country.csv", 'wb'))

df.drop(labels = ["Unnamed: 0.1"], axis=1, inplace=True)

df.drop("Unnamed: 0", axis=1, inplace=True)

display(df.head(10))



print("Number of matches in dataset", len(df),"\n")

# Exploring null values

null_cols = df.isnull().any()

print("number of columns with null values", len(null_cols[null_cols]), "\n")

print("Columns with null values:", list(null_cols[null_cols].keys()), "\n")



teams = list(set(df["Team 1"]).union(set(df["Team 2"])))

print("Total number of teams", len(teams))

print(teams)



print_runtime("Cell 1 completed", stime)
print(df.dtypes)
count = 0

for date in tqdm(df["Match Date"]):

    if not re.match("[A-Za-z]{3}\s\d{1,2},\s\d{4}", date):

        print(date, "<===>" ,re.sub("\d{1,2}-","", date))

        count = count + 1

        if count > 10:

            break
df["Match Date"] = df["Match Date"].str.replace("\d{1,2}-","")
print("Before conversion dtype:", df["Match Date"].dtype)

df["Match Date"] = pd.to_datetime(df["Match Date"], format='%b %d, %Y')

print("After conversion dtype:", df["Match Date"].dtype)

display(df.head(10))
if len(null_cols) > 0:

    df_na = df[df.isnull().any(axis=1)]

    print("Unique values in 'Winner column'", df_na["Winner"].unique(),"\n")

    print("checking for matches where Margin is null but Winner has some value--\n")

    display(df_na[(df_na["Winner"] != "no result") & (df_na["Winner"] != "tied")])

    

    print("\nNumber of records with null values", len(df_na), "which is", 

          np.round((len(df_na)/len(df))*100, 2), "% of total data\n")

    null_team_counts = add_feature_value_fequencies(df_na, "Team 1", "Team 2")

    # print(null_team_counts)
if null_team_counts is not None:

    null_team_counts = dict(null_team_counts)

    plt.figure(figsize=(20, 10))

    plt.bar(breakline_labels(list(null_team_counts.keys())), 

            null_team_counts.values(), width=0.5, align='center', 

            color=generate_colors_by_magnitude(null_team_counts.values()))

    plt.title("Team with undecided matches")

    plt.show()

df_temp = df_na[(df_na["Team 1"]=="India") | (df_na["Team 2"]=="India")]



# To find which team has highest undecided result with India we can swap 'Team 1'

# and 'Team 2' values such that 'Team 1' has only "India" string, so that we 

# do a groupby and find frequency of each team in 'Team 2'

pd.options.mode.chained_assignment = None # suppress copy warning for this op



bool_mask = df_temp["Team 2"] == "India"

swap_val = df_temp.loc[bool_mask]["Team 1"]

df_temp.loc[bool_mask, ["Team 1"]] = "India"

df_temp.loc[bool_mask, ["Team 2"]] = swap_val

opponents = df_temp["Team 2"].value_counts()



pd.options.mode.chained_assignment = "warn"  # reset copy warning for this op



plt.figure(figsize=(10, 7))

pie(opponents, labels=list(opponents.keys()), explode=opponents*0.004,

    colors=generate_colors_by_magnitude(opponents), shadow=True, autopct='%1.1f%%')

plt.title("Teams that played with India with most undecided results")

plt.show()
df_temp = df[df["Winner"] == df["HostCountry"]]

host_winners = df_temp["Winner"].value_counts()



plt.figure(figsize=(10, 7))

plt.bar(breakline_labels(list(host_winners.keys())), 

        host_winners, width=0.5, align='center', 

        color=generate_colors_by_magnitude(host_winners))

plt.xticks(rotation='vertical')

plt.title("Team wins on home grounds")

plt.show()
zero_home_winners = list(set(teams).difference(set(host_winners.keys())))

print(zero_home_winners)
display(df[ df["Team 1"].isin(zero_home_winners) | df["Team 2"].isin(zero_home_winners) ].head(10))
# teams that are not present in HostCountry

differing_team_country = list(set(teams).difference(set(df["HostCountry"]))) 

# HostCountry values that are not present in teams

differing_team_country2 = list(set(df["HostCountry"]).difference(set(teams)))



print("Teams =",differing_team_country) # 15

print("HostCountries =", differing_team_country2) # 20



# most_similar_to_given
teams_to_drop = ["Asia XI", "East Africa", "Nepal", 'ICC World XI', 'Africa XI']

for d in teams_to_drop:

    try:

        differing_team_country.remove(d)

    except:

        print(d, "not in list")

    

print(differing_team_country)
from gensim.models import Word2Vec

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec



stime = time.time()

glove_file = "glove.6B.200d.txt"

print("converting Glove to Word2Vec")

glove2word2vec(glove_input_file=os.path.join(DRIVE_INPUT_PATH, glove_file), word2vec_output_file="gensim_glove_vectors.txt")

print("loading Word2Vec")

glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

print_runtime("loading done", stime)  # took approx. 1.5 min
word1 = "america"

word2 = "usa"



print("Similarity between", word1, "&",word2, "is", glove_model.similarity(word1, word2))



word1 = "america"

word2 = "jupiter"



print("Similarity between", word1, "&",word2, "is", glove_model.similarity(word1, word2))



word1 = "man"

word2 = "human"



print("Similarity between", word1, "&",word2, "is", glove_model.similarity(word1, word2))



word1 = "man"

word2 = "kangaroo"



print("Similarity between", word1, "&",word2, "is", glove_model.similarity(word1, word2))
scores = {}

for w1 in differing_team_country:  # teams

    w1s = w1.replace('.', '').lower().split()



    # print("w1s",w1s)



    for w2 in differing_team_country2:  # countries

        w2s = w2.replace('.', '').lower().split()

        sim = 0

        for x in w1s:

            for y in w2s:

                try:

                    res = glove_model.similarity(x, y)

                    # print(x,"<-->", y, res)

                    sim = sim + res

                except:

                    # print(w, "not found")

                    pass

        scores[(w1, w2)] = sim

        # print((w1, w2), scores[(w1,w2)])



for team in differing_team_country:

    list_team_scores = []

    list_team_hosts = []

    for t in scores:

        if t[0] == team:

            list_team_scores.append(scores[t])

            list_team_hosts.append(t[1])

    _ = sorted(range(len(list_team_scores)),

               key=list_team_scores.__getitem__, reverse=True)

    for i in _[:5]:

        print (team, '&', list_team_hosts[i], '=', list_team_scores[i])

    print('-----------------------------------------------------------------------')

values = ["U.A.E.", "England", "South Africa", "P.N.G.", "U.S.A.", "Hong Kong"]

to_replace = ["United Arab Emirates", "United Kingdom",

             "RSA", "Papua New Guinea", "USA", "PRC"]

df["HostCountry"].replace(to_replace, values, inplace=True)

df["HostCountry"].replace( ["Saint Lucia", "Cayman Islands", 

                           "Jamaica", 'Saint Vincent and the Grenadines', 

                            "Trinidad Tobago", "Barbados", 

                            "Saint Kitts and Nevis", "Dominica"], "West Indies",

                          inplace=True)

print("Unique values in HostCountry column now--\n",df["HostCountry"].unique())
df_temp = df[df["Winner"] == df["HostCountry"]]

host_winners = df_temp["Winner"].value_counts()

host_winners.sort_values(inplace=True, ascending=False)



df_temp = df[df["Winner"] != df["HostCountry"]]

guest_winners = df_temp["Winner"].value_counts()

guest_winners.sort_values(inplace=True, ascending=False)



inds = list(set(host_winners.keys()).intersection(set(guest_winners.keys())))

# print(inds)



hw_vals = host_winners[inds]

gw_vals = guest_winners[inds]

# print("hw_vals\n", hw_vals)

# print("gw_vals\n", gw_vals)



width = 0.35

plt.figure(figsize=(15, 10))

plt.bar(np.arange(len(inds)), #breakline_labels(list(host_winners.keys())) 

        hw_vals[inds], width=width, align='center', 

        color='g')





plt.bar(np.arange(len(inds))+width, 

        gw_vals[inds], width=width, align='center', 

        color='b')



# print(len(host_winners), len(guest_winners))

plt.xticks(np.arange(len(inds))+0.5/2, inds)

plt.title("Team wins on home grounds vs overseas")

plt.ylabel("Wins")

plt.xticks(rotation='vertical')

plt.legend(["Home", "Overseas"])

plt.show()



# print(host_winners["Pakistan"])
df["year"] = df["Match Date"].apply(lambda x: x.year)

# display(df[["Match Date", "year"]].head())
matches_per_year = df["year"].value_counts()

matches_per_year.sort_index(inplace=True)



plt.figure(figsize = (10, 7))

plt.title("Number of ODIs played per year")

plt.plot(matches_per_year.keys()[:-1], matches_per_year[:-1])  # leave 2019 

plt.xlabel("Years")

plt.ylabel("Matches")

plt.show()
df_temp = df[["Team 1", "Team 2"]]

print("This what origin data looks like. To be able to count number of times each pair of teams have played I will sort the order of team columns.")

display(df_temp.head())

df_temp.apply(lambda row: swap_columns(row, "Team 1", "Team 2"), axis=1)

print("Sorted values on axis = 1")

display(df_temp.head())



# now we can groupby Team 1 and Team 2

df_temp = df_temp.groupby(["Team 1", "Team 2"]).size().reset_index(name="count")  # [["Team 1", "Team 2"]].aggregate("count")

df_temp.sort_values("count", inplace=True, ascending=False)

df_temp = df_temp[df_temp["count"] > 50]



plt.figure(figsize=(15, 10))

x_ticks = ["{} vs {}".format(t1, t2) for t1, t2, in zip(shorten_labels(df_temp["Team 1"]), shorten_labels(df_temp["Team 2"]))]

plt.bar(x_ticks, df_temp["count"], color=generate_colors_by_magnitude(df_temp["count"]))

plt.xticks(rotation='vertical')

plt.xlabel("Team pairs")

plt.ylabel("Number of matches played")

plt.show()