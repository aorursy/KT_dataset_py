# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas_profiling import ProfileReport

from enum import Enum

from collections import Counter
import spacy

spacy.prefer_gpu()
FILE1 = "/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv"

FILE2 = "/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv"
df1 = pd.read_csv(FILE1)

df2 = pd.read_csv(FILE2)
df1.columns = ['id','df1_title','cast','crew']

merged = df2.merge(df1, on='id')
merged[['title','df1_title']].head()
# verify the merge was okay

should_be_true = merged[['title','df1_title']].apply(lambda x: x[0]==x[1], axis='columns').all()

assert should_be_true, "uh oh"
simple_df = merged[[

    "title",

    "cast",

    "crew",

    "budget", 

    "genres", 

    "overview", 

    "production_companies", 

    "production_countries", 

    "original_language", 

    "original_title", 

    "keywords",

]]
simple_df.head()
def to_df(value):

    assert type(value) == str

    lst = eval(value)  # ASSUME: value is a string repr of list containing dicts "[{}]"

    assert type(lst) == list

    if len(lst) > 0:

        assert type(lst[0]) == dict

    return pd.DataFrame(lst)



def with_value_as_df(fn=lambda df:df):

    def _fn(value):

        # ASSUME: cast dict can be converted into DataFrame

        df = to_df(value)

        return fn(df)

    return _fn
simple_df['cast_dfs'] = simple_df['cast'].apply(to_df)

simple_df['crew_dfs'] = simple_df['crew'].apply(to_df)


class MESSAGE(Enum):

    SUCCESS = "MESSAGE.SUCCESS"

    ORDER_NOT_IN_CAST_DICT = "MESSAGE.ORDER_NOT_IN_CAST_DICT"

    MORE_THAN_ONE_RESULT_FOUND = "MESSAGE.MORE_THAN_ONE_RESULT_FOUND_FOR_CAST"

    EMPTY_CAST = "MESSAGE.EMPTY_CAST"

    NO_JOB_CREW = "MESSAGE.NO_JOB_CREW"

    EMPTY_CREW = "MESSAGE.EMPTY_CREW"

    UNEXPECTED = "MESSAGE.UNEXPECTED"

    NO_DIRECTOR = "MESSAGE.NO_DIRECTOR"

    TOO_MANY_DIRECTORS = "MESSAGE.TOO_MANY_DIRECTORS"



def get_cast_at_orders(order_nums=[], ret_fun=lambda x,msg:x):

    def fn(df, order_num):

        if 'order' not in df:

            return ret_fun(df, MESSAGE.ORDER_NOT_IN_CAST_DICT)

        cast = df[df['order'] == order_num]

        if len(cast) <= 0:

            return ret_fun(cast, MESSAGE.EMPTY_CAST)

        if len(cast) >= 2:

            return ret_fun(cast, MESSAGE.MORE_THAN_ONE_RESULT_FOUND)

        return ret_fun(cast, MESSAGE.SUCCESS)

    def fn2(df):

        return tuple(fn(df, num) for num in order_nums)

    return fn2



def with_err_handling(ret_fun=lambda df:df):

    def fn(df, msg):

        if msg == MESSAGE.ORDER_NOT_IN_CAST_DICT:

            return MESSAGE.ORDER_NOT_IN_CAST_DICT.value

        elif msg == MESSAGE.EMPTY_CAST:

            return MESSAGE.EMPTY_CAST.value

        elif msg == MESSAGE.MORE_THAN_ONE_RESULT_FOUND:

            return MESSAGE.MORE_THAN_ONE_RESULT_FOUND.value

        elif msg == MESSAGE.SUCCESS:

            # there is exactly 1 cast member in df

            return ret_fun(df)

        else:

            assert False, "unexpected"    

    return fn



get_name = with_err_handling(lambda df: df['name'].values[0])

get_char = with_err_handling(lambda df: df['character'].values[0])

get_name_and_char = with_err_handling(lambda df: (df['name'].values[0], df['character'].values[0]))
print(MESSAGE.EMPTY_CREW.value)

print(MESSAGE.EMPTY_CREW)

print(str(MESSAGE.EMPTY_CREW))

MESSAGE.SUCCESS
print(simple_df.columns)



print(type(simple_df['cast_dfs'][0]))



print(simple_df['cast_dfs'][0].columns)
NUMS = [0, 1, 2, 3, 4]

tuple_series  = simple_df['cast_dfs'].apply(get_cast_at_orders(NUMS, get_name_and_char))

#

simple_df['order_0_cast_name'] = tuple_series.apply(lambda x: x[0][0])

simple_df['order_1_cast_name'] = tuple_series.apply(lambda x: x[1][0])

simple_df['order_2_cast_name'] = tuple_series.apply(lambda x: x[2][0])

simple_df['order_3_cast_name'] = tuple_series.apply(lambda x: x[3][0])

simple_df['order_4_cast_name'] = tuple_series.apply(lambda x: x[4][0])

# 

simple_df['order_0_cast_char'] = tuple_series.apply(lambda x: x[0][1])

simple_df['order_1_cast_char'] = tuple_series.apply(lambda x: x[1][1])

simple_df['order_2_cast_char'] = tuple_series.apply(lambda x: x[2][1])

simple_df['order_3_cast_char'] = tuple_series.apply(lambda x: x[3][1])

simple_df['order_4_cast_char'] = tuple_series.apply(lambda x: x[4][1])
print(simple_df.columns)



print(type(simple_df['crew_dfs'][0]))



print(simple_df['crew_dfs'][0].columns)
def get_director(df):

    if 'job' not in df:

        return MESSAGE.NO_JOB_CREW.value

    if type(df) != pd.core.frame.DataFrame:

        return MESSAGE.UNEXPECTED.value + "_expected_dataframe"

    if len(df) <= 0:

        return MESSAGE.EMPTY_CREW.value + "_zero_len"

    if df.size <= 0:

        return MESSAGE.EMPTY_CREW.value + "_zero_size"

    director_series = df[df['job'] == 'Director']

    if type(director_series['name']) != pd.core.series.Series:

        return MESSAGE.UNEXPECTED.value + "_expected_series"

    director_values = director_series.values

    dir_vals_len = len(director_values)

    if (dir_vals_len <= 0):

        return MESSAGE.NO_DIRECTOR.value

    if (type(director_values[0]) != np.ndarray):

        return MESSAGE.UNEXPECTED.value + "_expected_ndarray_for_director"

    if (dir_vals_len >= 2):

        return MESSAGE.TOO_MANY_DIRECTORS.value

    name = director_values[0][-1]

    return name
simple_df['director'] = simple_df['crew_dfs'].apply(get_director)
x = simple_df['crew_dfs'][0]

y = x[x['job'] == 'Director']

print(type(y))

print(y.size)

print(len(y))

print(y)

n = y['name']

print(type(n))

print(n.size)

print(len(n))

print(n)

print(n.values)

print(type(n.values))

print(type(n.values[0]))
simple_df.columns
def get_genre(x):

    try:

        lst = eval(x)

        return tuple(d['name'] for d in lst)

    except:

        return MESSAGE.UNEXPECTED.value + "_bad_genre_data"



simple_df['genre_tuples'] = simple_df['genres'].apply(get_genre)
output_df = simple_df[['title', 'overview', 'director', 'genre_tuples', 'budget',

                       'order_0_cast_name', 'order_1_cast_name', 'order_2_cast_name', 

                       'order_3_cast_name', 'order_4_cast_name', 'order_0_cast_char', 

                       'order_1_cast_char', 'order_2_cast_char', 'order_3_cast_char', 

                       'order_4_cast_char']]
nlp = spacy.load('en_core_web_lg')
def get_five_common_words(overview):

    if type(overview) != str:

        return MESSAGE.UNEXPECTED.value + "_expected_str_overview"

    doc = nlp(overview)

    # all tokens that arent stop words or punctuations

    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

    # five most common tokens

    word_freq = Counter(words)

    common_words = word_freq.most_common(5)

    return tuple(x[0] for x in common_words)



def get_five_common_nouns(overview):

    if type(overview) != str:

        return MESSAGE.UNEXPECTED.value + "_expected_str_overview"

    doc = nlp(overview)

    # noun tokens that arent stop words or punctuations

    nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]

    # five most common noun tokens

    noun_freq = Counter(nouns)

    common_nouns = noun_freq.most_common(5)

    return tuple(x[0] for x in common_nouns)



def get_five_common_verbs(overview):

    if type(overview) != str:

        return MESSAGE.UNEXPECTED.value + "_expected_str_overview"

    doc = nlp(overview)

    # noun tokens that arent stop words or punctuations

    verbs = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "VERB"]

    # five most common verbs tokens

    noun_freq = Counter(verbs)

    common_verbs = noun_freq.most_common(5)

    return tuple(x[0] for x in common_verbs)





def get_entities(overview):

    if type(overview) != str:

        return MESSAGE.UNEXPECTED.value + "_expected_str_overview"

    doc = nlp(overview)

    return tuple(ent.text for ent in doc.ents)
output_df['common_words'] = output_df['overview'].apply(get_five_common_words)
output_df['common_nouns'] = output_df['overview'].apply(get_five_common_nouns)
output_df['common_verbs'] = output_df['overview'].apply(get_five_common_verbs)
output_df['entities'] = output_df['overview'].apply(get_entities)
output_df.columns
bad_director_data = output_df['director'].str.startswith('MESSAGE')

bad_genre_data = output_df['genre_tuples'].astype(str).str.startswith('MESSAGE')

bad_data_1 = output_df['order_0_cast_name'].astype(str).str.startswith('MESSAGE')

bad_data_2 = output_df['order_1_cast_name'].astype(str).str.startswith('MESSAGE')

bad_data_3 = output_df['order_2_cast_name'].astype(str).str.startswith('MESSAGE')

bad_data_4 = output_df['order_3_cast_name'].astype(str).str.startswith('MESSAGE')

bad_data_5 = output_df['order_4_cast_name'].astype(str).str.startswith('MESSAGE')

bad_data_6 = output_df['order_1_cast_char'].astype(str).str.startswith('MESSAGE')

bad_data_7 = output_df['order_2_cast_char'].astype(str).str.startswith('MESSAGE')

bad_data_8 = output_df['order_3_cast_char'].astype(str).str.startswith('MESSAGE')

bad_data_9 = output_df['order_4_cast_char'].astype(str).str.startswith('MESSAGE')

bad_data_10 = output_df['common_words'].astype(str).str.startswith('MESSAGE')

bad_data_11 = output_df['common_nouns'].astype(str).str.startswith('MESSAGE')



bad = (

    bad_director_data |

    bad_genre_data | 

    bad_data_1 |

    bad_data_2 | 

    bad_data_3 | 

    bad_data_4 | 

    bad_data_5 | 

    bad_data_6 | 

    bad_data_7 | 

    bad_data_8 | 

    bad_data_9 | 

    bad_data_10 | 

    bad_data_11

)
good_output_df = output_df[~bad]
good_output_df['genre_tuples_str'] = good_output_df['genre_tuples'].astype(str)

good_output_df['common_words_str'] = good_output_df['common_words'].astype(str)

good_output_df['entities_str']     = good_output_df['entities'].astype(str)

good_output_df['common_nouns_str'] = good_output_df['common_nouns'].astype(str)

good_output_df['common_verbs_str'] = good_output_df['common_verbs'].astype(str)
good_output_df['most_common_word'] = good_output_df['common_words'].apply(lambda x: x[0])
good_output_df['first_genre'] = good_output_df['genre_tuples'].apply(lambda x: x[0] if len(x) else "")
good_output_df.head()
good_output_df.to_csv("output.csv")
profile1 = ProfileReport(good_output_df, title='output_df Pandas Profiling Report', html={'style':{'full_width':True}})
profile1.to_notebook_iframe()
print("done")