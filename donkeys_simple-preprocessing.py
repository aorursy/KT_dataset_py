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
df_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

df_test.head()
target = df_train["target"]
df_train.drop("target", inplace=True, axis=1)
df_train["train_test"] = 1

df_test["train_test"] = 0
df_full = pd.concat([df_train, df_test], axis=0)

df_full.shape
binaries = [col for col in df_full.columns if "bin_" in col]

binaries
nominals = [col for col in df_full.columns if "nom_" in col]

nominals
for nom_col in nominals:

    unique_vals = df_full[nom_col].unique()

    print(f"nominal: {nom_col}, unique_vals={len(unique_vals)}")
for nom_col in nominals:

    unique_vals = df_full[nom_col].unique()

    unique_vals.sort()

    print(f"nominal: {nom_col}, unique_vals={len(unique_vals)}")

    if len(unique_vals) < 100:

        for val in unique_vals:

            print(f'"{val}" ', end='')

    print()

    print()
ordinals = [col for col in df_full.columns if "ord_" in col]

ordinals
for ord_col in ordinals:

    unique_vals = df_full[ord_col].unique()

    print(f"ordinal: {ord_col}, unique_vals={len(unique_vals)}")
for ord_col in ordinals:

    unique_vals = df_full[ord_col].unique()

    if ord_col == "ord_5":

        unique_vals = sorted(unique_vals, key=str.swapcase)

    else:

        unique_vals.sort()

    print(f"ordinal: {ord_col}, unique_vals={len(unique_vals)}")

    for val in unique_vals:

        print(f'"{val}" ', end='')

    print()

    print()

kaggle_level = {'Novice':1,'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}

df_full['ord_1'] = df_full['ord_1'].map(kaggle_level)
temperature = {'Freezing':1,'Cold':2,'Warm':3,'Hot':4,'Boiling Hot':5,'Lava Hot':6}

df_full['ord_2'] = df_full['ord_2'].map(temperature)
letter_cols = ["ord_3", "ord_4", "ord_5"]

for ord_col in letter_cols:

    mapping = {}

    unique_vals = df_full[ord_col].unique()

    if ord_col == "ord_5":

        continue

#        unique_vals = sorted(unique_vals, key=str.swapcase)

    else:

        unique_vals.sort()

    index = 0

    for val in unique_vals:

        mapping[val] = index

        index += 1

    df_full[ord_col] = df_full[ord_col].map(mapping)
import string



string_chars = string.ascii_letters

string_chars
print(len(string_chars))

chars = int(len(string_chars)/2)

print(chars)
index = 1

char_map = {}

#string_chars = string_chars[::-1]

for val in string_chars:

    char_map[val] = index

    index += 1



for k,v in char_map.items():

    print(f'{k}={v} ', end='')

def calc_org5(o5_chars):

    count_chars = len(char_map)

    c1 = o5_chars[0]

    c2 = o5_chars[1]

    #print(c1)

    val1 = char_map[c1]*count_chars

    val1 += char_map[c2]

    return val1



df_full["ord_5"] = df_full["ord_5"].map(lambda x: calc_org5(x))

df_full.head()
cyclicals = ["day", "month"]

cyclicals
for cyc_col in cyclicals:

    unique_vals = df_full[cyc_col].unique()

    print(f"cyclical: {cyc_col}, unique_vals={len(unique_vals)}")

#    print(df_train[nom_col].unique())
from sklearn import preprocessing



label_encoders = {}

to_label_encode = nominals + ["bin_3", "bin_4"]
for col in to_label_encode:

    le = preprocessing.LabelEncoder()

    le.fit(df_full[col])

    df_full[col] = le.transform(df_full[col])

    label_encoders[col] = le
for col in [binaries + nominals + cyclicals]:

    df_full[col] = df_full[col].astype("category")
for col in ordinals:

    print(col)

    df_full[col] = df_full[col].astype("int16")
df_train = df_full[df_full["train_test"] == 1].copy()

df_test = df_full[df_full["train_test"] == 0].copy()
df_train.drop("train_test", axis=1, inplace=True)

df_train.columns
df_test.drop("train_test", axis=1, inplace=True)

df_test.columns
df_train["target"] = target.astype(np.uint8)

df_train.to_csv("df_train.csv")

df_test.to_csv("df_test.csv")
df_train.dtypes
import pickle



def pickle_dataframe(df, filename):

    filehandler = open(f"{filename}.pkl","wb")

    pickle.dump(df,filehandler)

    filehandler.close()



pickle_dataframe(df_train, "df_train")

pickle_dataframe(df_test, "df_test")