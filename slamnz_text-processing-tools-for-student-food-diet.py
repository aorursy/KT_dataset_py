from pandas import read_csv

data = read_csv("../input/food_coded.csv")
textual_features = []

for feature in data.columns:

    if feature in ["GPA","weight"]: continue

    if data[feature].dtype == object: textual_features += [feature]
data[textual_features].head()
# Lowercase text

str.lower("")



# Uppercase text

str.upper("")



# Capitalize first letter of text

str.capitalize("")



# Join a list of string into a single string

", ".join(["chocolate", "banana", "cookie"])



# Remove abritrarily chosen character from first and last index of a string

" he llo ".strip(" ")



# Replace specific characters or substrings with chosen subsitute

"hello this is -remove this please- replace text".replace("-remove this please- ", "")



# Are all characters in string only alphabetical or numerical?

"text".isalnum()

"text119082329112".isalnum()



# Are all characters in string digits only?

"text1".isdigit()



pass
from nltk import word_tokenize, FreqDist, bigrams, everygrams, trigrams, pos_tag

from nltk.corpus import stopwords



def token_count(series):

    return series.apply(lambda x: len(word_tokenize(str(x))))

    

def character_count(series):

    return series.apply(lambda x: str(x).replace(" ", "")).apply(lambda x: len(x))



def mean_characters_per_word(series):

    return character_count(series).divide(word_count(series))



def unique_vocabulary_count(series):

    return series.apply(str).apply(lambda x: len(set([word for word in word_tokenize(x)])))



def lexical_diversity(series):

    return series.apply(str).apply(lambda x: len(set([word for word in word_tokenize(x) if word not in stopwords.words("english")])) / len([word for word in word_tokenize(x) if word not in stopwords.words("english")]))



def word_is_present(series, word):

    return series.apply(str).apply(str.lower).apply(lambda x: 1 if word in x else 0)



def item_count(series):

    return series.apply(str).apply(lambda x: len(x.replace("/", ",").split(",")))



def manual_clean_fuzzy_string(series, base, derivatives):

    

    cleaned = []

    

    for row in series:

        

        row = str(row)

        

        for d in derivatives:

            if d in row:

                row.replace(base,d)

        

        cleaned += [row]

    

    cleaned = Series(cleaned)

    cleaned.name = "cleaned_" + series.name

    

    return cleaned



def add_word_presence_features(data,text_feature, word_list):

    

    for w in word_list: 

        key = text_feature + "/" + w

        dictionary = { key : word_is_present(data[text_feature],w)}

        data[key] = word_is_present(data[text_feature],w)

        

    return data



from collections import Counter

from pandas import Series



def parts_of_speech_counter(series):

    

    adjective_count = []

    noun_count = []

    verb_count = []

    

    for row in series:

        row = str(row)

        tokens = set(word_tokenize(row))

        tuples = pos_tag(tokens, tagset="universal")

        counts = Counter([t[1] for t in tuples])

        

        adjective_count += [counts["ADJ"]]

        noun_count += [counts["NOUN"]]

        verb_count += [counts["VERB"]]

        

    return Series(adjective_count, name = series.name + " / Adjective Count"), Series(noun_count, name = series.name + " / Noun Count"), Series(verb_count, name = series.name + " / Verb Count")
def find_most_common(series):

    series = series.apply(str).apply(str.lower).tolist()

    raw_text = " ".join(series)

    tokens = word_tokenize(raw_text)

    tokens = [token for token in tokens if token not in stopwords.words("english") + [",",".","&"]]

    grams = everygrams(tokens, max_len=1)

    frequencies = FreqDist(grams)

    return frequencies.most_common(60)
i = iter(textual_features)
selected_comfort_foods = ["ice", "pizza", "chocolate","chips","mac", "candy","pasta","soup", "cake", "fries", "chinese", "cheese"]
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_comfort_food_reasons = ["bored","sad","stress","comfort","anger","happiness"]
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_diet_current = ["healthy","veg","diet","fruit","meat","carbs","protein"]
feature = next(i)

print(feature)

find_most_common(data[feature])
feature = next(i)

print(feature)

find_most_common(data[feature])
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_fav_cuisine = ["italian", "mexican", "chinese", "american"]
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_food_childhood = ["chicken", "pizza", "cheese", "pasta", "spaghetti", "mac", "steak", "potatoes", "tacos"]
feature = next(i)

print(feature)

find_most_common(data[feature])
feature = next(i)

print(feature)

find_most_common(data[feature])
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_meals_dinner_friend = ["pizza", "steak", "pasta", "rice", "lasagna", "spaghetti", "tacos", "chicken", "salad", "grilled", "steak"]
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_mother_profession = ["teacher","secretary"]
feature = next(i)

print(feature)

find_most_common(data[feature])
selected_sports = ["hockey", "soccer", "basketball", "softball", "volleyball"]
feature = next(i)

print(feature)

sample = find_most_common(data[feature])
sample[0]