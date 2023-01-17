from hashlib import md5

import random

import re

import requests

from IPython.display import display

from matplotlib_venn import venn2

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



sns.set()
labels_df = pd.read_csv("../input/cute-cats-and-dogs-from-pixabaycom/labels.csv")

test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")



labels_df.head()
test["Name"] = test["Name"].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', str(x)))
hash_cols = ['Name','Type','Quantity','Vaccinated','Quantity',

             'Dewormed','Sterilized','Gender','MaturitySize','FurLength',

             'Color1','Color2','Color3','Health','Fee']

pet_info = test[hash_cols].apply(lambda cols: "".join(str(x) for x in cols), axis=1)

display(pet_info.head())

print("% unique:", 1 - pet_info.duplicated().mean())
hashed_pet_info = pet_info.apply(lambda x: md5(x.encode('utf-8')).hexdigest()[:-1])

hashed_pet_info.head()
hash_dict = dict()



for x in list(labels_df["id"]):

    if x[-1].isnumeric():

        hash_dict[x[:-1]] = int(x[-1]) // 2



# just a sample of the hash dict keys

rand_keys = np.random.choice(list(hash_dict.keys()), size=10)

{key: hash_dict[key] for key in rand_keys}
plt.figure(figsize=(14, 7))

venn2([

    set(hashed_pet_info), 

    set(hash_dict.keys())

], set_labels=["Pet Hashes", "External ID Column"])
test["AdoptionSpeed"] = hashed_pet_info.apply(lambda x: hash_dict.get(x, 2))

test[["PetID", "AdoptionSpeed"]].to_csv("submission.csv", index=False)

test["AdoptionSpeed"].head()
plt.figure(figsize=(14, 7))

def str_to_num(x):

    try:

        return int(x)

    except ValueError:

        return np.nan



raw_nums = labels_df["id"].str[-1].apply(str_to_num)

raw_nums.dropna().plot.hist(bins=19)

plt.title("Last digit distribution of cheating md5 hashes")

print()
import string



def get_random_string(minlen=5, maxlen=20):

    """Generate a random string of fixed length """

    letters = string.ascii_lowercase + string.digits

#     letters = string.ascii_lowercase[:3] # also when only using a small subset of letters so it does not depend on the randomness of the input strings

    length = random.randint(minlen, maxlen)

    return ''.join(random.choice(letters) for i in range(length))



random_hashes = pd.Series([md5(get_random_string().encode("utf-8")).hexdigest() for _ in range(100000)])

plt.figure(figsize=(14, 7))

(random_hashes.str[-1].apply(str_to_num)).dropna().plot.hist(bins=19)

plt.title("Last digit distribution of md5 hashes of random strings.")

print()
def majority_voting(df, cols):

    df["tmp"] = df[cols].mode(axis=1)[0]

    return proc(df)
cols_ = ['Name','Type','Quantity','Vaccinated','Quantity',

         'Dewormed','Sterilized','Gender','MaturitySize','FurLength',

         'Color1','Color2','Color3','Health','Fee']



def proc(df):

    df["Name"] = df["Name"].apply(clean_name)

    col_to_str(df, cols_)

    dic = get_dict(labels_df)

    res = process(df, dic)

    df.drop(["str","tmp"],axis=1,inplace=True)

    return res
def clean_name(x):

    if str(x)=="nan":

        return "nan"

    return re.sub('[^A-Za-z0-9]+','',x)
def col_to_str(df, cols):

    df["str"] = ""

    for c in cols:

        df["str"] += df[c].astype(str)

    df["str"] = df["str"].str.replace(" ",'')
def get_dict(df):

    d = dict()

    for x in list(df.id):

        if x[-1].isnumeric():

            d[x[:-1]]=int(x[-1])//2

    return d
def process(df, d):

    res, i = [], -1

    for x,y in zip(df["str"], df["tmp"]):

        k = md5(x.encode('utf-8')).hexdigest()[:-1]

        res.append(d[k]) if k in d and i%10==0 else res.append(y)

        i+=1

    return res
test["preds"] = -1

sub_preds = np.array(majority_voting(test, ["preds"]))

(sub_preds != -1).sum() # expected: 10% of 3500