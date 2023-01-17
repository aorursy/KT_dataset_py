import numpy as np

import pandas as pd

from os.path import join

from os import listdir

import time

import string
DATA_DIR = "../input/sentences"

OUTPUT_DIR = "."

FILENAMES = listdir(DATA_DIR)

print(FILENAMES)
def remove_puncts_from_numpy_of_strings(arr):

    assert arr.dtype.type == np.str_, "please only input numpy arrays with strings"

    for c in string.punctuation:

        arr = np.char.replace(arr, c, "")

    return arr



def assert_no_puncts_in_numpy_of_strings(arr):

    assert arr.dtype.type == np.str_, "please only input numpy arrays with strings"

    for c in string.punctuation:

        assert np.char.find(arr, c).all(-1), "found punctuation {}".format(c)



    assert len(arr[arr == ""]) == 0, "whoops, found an empty string"

    return arr



def numpy_string_array_but_without_empty_strings(arr):

    return arr[arr != ""]
FILE_PATH = join(DATA_DIR, "tatoeba_sentences.csv")



df = pd.read_csv(FILE_PATH, sep='\t', encoding="utf-8", header=None)



df.columns = ['num','lang', 'sent']



df = df[df['lang'] == 'eng']['sent']
df.head()
df1 = df[:20000]

df2 = df[20000:60000]

df3 = df[60000:100000]

df4 = df[100000:500000]

df5_a = df[500000:600000]

df5_b = df[600000:700000]

df5_c = df[700000:800000]

df5_d = df[800000:900000]

df5_e = df[900000:1000000]

df6_a = df[1000000:1100000]

df6_b = df[1100000:1200000]

df6_c = df[1200000:]
print(len(df), "= sum of(", len(df1),len(df2),len(df3),len(df4),

      len(df5_a),len(df5_b),len(df5_c),len(df5_d),len(df5_e),

      len(df6_a),len(df6_b), len(df6_c), ")")
df1 = remove_puncts_from_numpy_of_strings(df1.unique().astype('str'))
df1 = numpy_string_array_but_without_empty_strings(df1)

assert_no_puncts_in_numpy_of_strings(df1)
df2 = remove_puncts_from_numpy_of_strings(df2.unique().astype('str'))
df2 = numpy_string_array_but_without_empty_strings(df2)

assert_no_puncts_in_numpy_of_strings(df2)
df3 = remove_puncts_from_numpy_of_strings(df3.unique().astype('str'))
df3 = numpy_string_array_but_without_empty_strings(df3)

assert_no_puncts_in_numpy_of_strings(df3)
# This line kills my computer

df4 = remove_puncts_from_numpy_of_strings(df4.unique().astype('str'))
df4 = numpy_string_array_but_without_empty_strings(df4)

assert_no_puncts_in_numpy_of_strings(df4)
# save the data

df_to_save = pd.DataFrame(np.concatenate([df1,df2,df3]))

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part1.csv")

df_to_save.to_csv(filepath)
# del df1

# del df2

# del df3
# save the data

df_to_save = pd.DataFrame(df4)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part2.csv")

df_to_save.to_csv(filepath)
# del df4
PART_NUM = 3

df5_a = remove_puncts_from_numpy_of_strings(df5_a.unique().astype('str'))

df5_a = numpy_string_array_but_without_empty_strings(df5_a)

assert_no_puncts_in_numpy_of_strings(df5_a)

# save the data

df_to_save = pd.DataFrame(df5_a)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 4



df5_b = remove_puncts_from_numpy_of_strings(df5_b.unique().astype('str'))

df5_b = numpy_string_array_but_without_empty_strings(df5_b)

assert_no_puncts_in_numpy_of_strings(df5_b)

# save the data

df_to_save = pd.DataFrame(df5_b)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 5

df5_c = remove_puncts_from_numpy_of_strings(df5_c.unique().astype('str'))

df5_c = numpy_string_array_but_without_empty_strings(df5_c)

assert_no_puncts_in_numpy_of_strings(df5_c)

# save the data

df_to_save = pd.DataFrame(df5_c)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 6

df5_d = remove_puncts_from_numpy_of_strings(df5_d.unique().astype('str'))

df5_d = numpy_string_array_but_without_empty_strings(df5_d)

assert_no_puncts_in_numpy_of_strings(df5_d)

# save the data

df_to_save = pd.DataFrame(df5_d)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 7

df5_e = remove_puncts_from_numpy_of_strings(df5_e.unique().astype('str'))

df5_e = numpy_string_array_but_without_empty_strings(df5_e)

assert_no_puncts_in_numpy_of_strings(df5_e)

# save the data

df_to_save = pd.DataFrame(df5_e)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 8

df6_a = remove_puncts_from_numpy_of_strings(df6_a.unique().astype('str'))

df6_a = numpy_string_array_but_without_empty_strings(df6_a)

assert_no_puncts_in_numpy_of_strings(df6_a)

# save the data

df_to_save = pd.DataFrame(df6_a)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 9

df6_b = remove_puncts_from_numpy_of_strings(df6_b.unique().astype('str'))

df6_b = numpy_string_array_but_without_empty_strings(df6_b)

assert_no_puncts_in_numpy_of_strings(df6_b)

# save the data

df_to_save = pd.DataFrame(df6_b)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
PART_NUM = 10

df6_c = remove_puncts_from_numpy_of_strings(df6_c.unique().astype('str'))

df6_c = numpy_string_array_but_without_empty_strings(df6_c)

assert_no_puncts_in_numpy_of_strings(df6_c)

# save the data

df_to_save = pd.DataFrame(df6_c)

df_to_save.columns = ['sentence']

filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

df_to_save.to_csv(filepath)
# PART_NUM = ___________

# ___________ = remove_puncts_from_numpy_of_strings(___________.unique().astype('str'))

# ___________ = numpy_string_array_but_without_empty_strings(___________)

# assert_no_puncts_in_numpy_of_strings(___________)

# # save the data

# df_to_save = pd.DataFrame(___________)

# df_to_save.columns = ['sentence']

# filepath = join(OUTPUT_DIR, "tatoeba_sentences-eng-part"+ str(PART_NUM) + ".csv")

# df_to_save.to_csv(filepath)