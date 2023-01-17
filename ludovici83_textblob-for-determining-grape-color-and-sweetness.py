# In this notebook we will first rank varieties of wine by sweetness 

# by determining the frequency of words associated with sweetness in the descriptions.



# Similarly we will determine the grape color for each variety by counting the 

# frequency of flavor terms associated with red wines (cherry, berry, blueberry, etc)

# and white wines (lime,lemon,apple,apricot,pineapple, etc)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from __future__ import unicode_literals

from textblob import TextBlob

import pandas as pd
wine = pd.read_csv("../input/winemag-data_first150k.csv",sep=",")
wine.head()
wine = wine.drop('Unnamed: 0',1)

wine = wine.drop_duplicates()
len(wine)
# keeping only varieties with at least 20 reviews

num_reviews = wine.groupby('variety').description.count().to_frame().reset_index()

num_reviews = num_reviews[num_reviews.description > 19]

frequent_varieties = num_reviews.variety.tolist()

wine_f = wine.loc[wine['variety'].isin(frequent_varieties)]
len(wine_f)
# we will do this by counting the frequency of words associated with sweetness 

# in the descriptions of each variety (the words will be "sweet","sweetness","sugar","sugary",

#"caramel", and "caramelized")

# converting to lowercase letters

wine['description'] = wine['description'].str.lower()
#example, calculate the frequency of sweetness-words in descriptions of Port wine

porto = wine[wine.variety == "Port"]

sweet_freq = []

for review in porto.description:

    review = TextBlob(review)

    num_sweet = review.words.count("sweet")

    num_sweetness = review.words.count("sweetness")

    num_sugar = review.words.count("sugar")

    num_sugary = review.words.count("sugary")

    num_caramel = review.words.count("caramel")

    num_caramelized = review.words.count("caramelized")

    total_sweet = num_sweet+num_sweetness+num_sugar+num_sugary+num_caramel+num_caramelized

    sweet_freq.append(total_sweet)



print(sum(sweet_freq)/len(sweet_freq))
# a function to locate the maximum element of a list

import numpy as np

def locate_max(list):

    biggest = np.max(list)

    return biggest, [index for index, element in enumerate(list) 

                      if biggest == element]
locate_max(sweet_freq)
porto.description.iloc[142]
#writing it as a function that assigns a sweetness-score based on a list of descriptions

def sweetness_score(descriptions):

    sweet_freq = []

    for review in descriptions:

        review = TextBlob(review)

        num_sweet = review.words.count("sweet")

        num_sweetness = review.words.count("sweetness")

        num_sugar = review.words.count("sugar")

        num_sugary = review.words.count("sugary")

        num_caramel = review.words.count("caramel")

        num_caramelized = review.words.count("caramelized")

        total_sweet = num_sweet+num_sweetness+num_sugar+num_sugary+num_caramel+num_caramelized

        sweet_freq.append(total_sweet)

    return float(sum(sweet_freq)/len(sweet_freq))
sweetness_score(porto.description)
#calculating the frequency of "sweetness-related" words for each variety of wine and saving 

#it in a list tuples of type (wine-variety,frequency)

sweet_list = []

for variety in wine_f.variety.unique():

    df_variety = wine_f[wine_f.variety == variety]

    sweet = sweetness_score(df_variety.description)

    sweet_list.append((variety,sweet))



# sorting from high sweeetness to low sweetness    

sorted_sweet_list = sorted(sweet_list, key=lambda x: -x[1])
# putting the list in dataframe format

df_sweetness = pd.DataFrame(sorted_sweet_list,columns=["variety","sweetness_score"])
# Barplot of the data

import matplotlib.pyplot as plt

plt.rcdefaults()

fig, ax = plt.subplots()

varieties = tuple(df_sweetness.variety.tolist())[:20]

varieties = [TextBlob(i) for i in varieties]

y_pos = np.arange(len(varieties))

performance = np.array(df_sweetness.sweetness_score)[:20]

error = np.random.rand(len(varieties))



plt.barh(y_pos, performance, align='center', alpha=0.5)

plt.yticks(y_pos, varieties)

plt.xlabel('Sweetness score')

plt.title('Wine-varieties by sweetness')

 

plt.show()
# we will keep only varieties containing at least 10 descriptions

num_reviews = wine.groupby('variety').description.count().to_frame().reset_index()

num_reviews = num_reviews[num_reviews.description > 9]

frequent_varieties = num_reviews.variety.tolist()

wine_ff = wine.loc[wine['variety'].isin(frequent_varieties)]
#writing it as a function that assigns a redness-score based on a list of descriptions

def redness_score(descriptions):

    red_freq = []

    for review in descriptions:

        review = TextBlob(review)

        n1 = review.words.count("cherry")

        n2 = review.words.count("berry")

        n3 = review.words.count("cherries")

        n4 = review.words.count("berries")

        n5 = review.words.count("red")

        n6 = review.words.count("raspberry")

        n7 = review.words.count("raspberries")

        n8 = review.words.count("blueberry")

        n9 = review.words.count("blueberries")

        n10 = review.words.count("blackberry")

        n11 = review.words.count("blackberries")

        total_red = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11

        red_freq.append(total_red)

    return float(sum(red_freq)/len(red_freq))
#writing it as a function that assigns a whiteness-score based on a list of descriptions

def whiteness_score(descriptions):

    white_freq = []

    for review in descriptions:

        review = TextBlob(review)

        n1 = review.words.count("lemon")

        n2 = review.words.count("lemons")

        n3 = review.words.count("lime")

        n4 = review.words.count("limes")

        n5 = review.words.count("peach")

        n6 = review.words.count("peaches")

        n7 = review.words.count("white")

        n8 = review.words.count("apricot")

        n9 = review.words.count("pear")

        n10 = review.words.count("apple")

        n11 = review.words.count("nectarine")

        n12 = review.words.count("orange")

        n13 = review.words.count("pineapple")

        total_white = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10+n11+n12+n13

        white_freq.append(total_white)

    return float(sum(white_freq)/len(white_freq))
red_types = []

for variety in wine_ff.variety.unique():

    df_variety = wine_ff[wine_ff.variety == variety]

    red = redness_score(df_variety.description)

    red_types.append((variety,red)) # a redness score is asigned to each variety
# putting it in dataframe format

color_classification =  pd.DataFrame.from_records(red_types,columns=["variety","redness_score"])

color_classification.head()
white_types = []

for variety in wine_ff.variety.unique():

    df_variety = wine_ff[wine_ff.variety == variety]

    white = whiteness_score(df_variety.description)

    white_types.append((variety,white)) # a whiteness score is asigned to each variety
white = pd.DataFrame.from_records(white_types,columns=["variety","whiteness_score"])
# merging the two dataframes, we have a dataframe which, for each variety provides a redness and whiteness score

color_classification = color_classification.merge(white,how='left',on='variety')
color_classification.sample(5)
# a function that compares the redness and whiteness score for each variety,

# and returns "red" if redness score is greater, "white" if whiteness score

# is greater, or "inconclusive" otherwise

def identify_color(redness,whiteness):

    if redness > whiteness:

        return "red"

    if redness < whiteness:

        return "white"

    else:

        return "inconclusive"
color_classification['color'] = np.vectorize(identify_color)(color_classification['redness_score'], color_classification['whiteness_score'])
color_classification.sample(9)
color_classification.color.value_counts()
reds = color_classification[color_classification.color == "red"]
reds["color_p"] = reds.redness_score-reds.whiteness_score
#checking dubious cases of red wines

reds.sort_values(by='color_p').head(10)
whites = color_classification[color_classification.color == "white"]
whites["color_p"] = whites.whiteness_score-whites.redness_score
#checking the more dubious cases of white wines

whites.sort_values(by='color_p').head(10)
# including color column in descriptions dataframe by merging

wine_f = wine_f.merge(color_classification,how='left',on="variety")
wine_f.sample(2)