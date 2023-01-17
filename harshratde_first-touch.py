# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from bs4 import BeautifulSoup

import re

import string
dataframes = {

    "cooking": pd.read_csv("../input/cooking.csv"),

    "crypto": pd.read_csv("../input/crypto.csv"),

    "robotics": pd.read_csv("../input/robotics.csv"),

    "biology": pd.read_csv("../input/biology.csv"),

    "travel": pd.read_csv("../input/travel.csv"),

    "diy": pd.read_csv("../input/diy.csv"),

}
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def stripTagsAndUris(x):

    if x:

        # BeautifulSoup on content

        soup = BeautifulSoup(x, "html.parser")

        # Stripping all <code> tags with their content if any

        if soup.code:

            soup.code.decompose()

        # Get all the text out of the html

        text =  soup.get_text()

        # Returning text stripping out all uris

        return re.sub(uri_re, "", text)

    else:

        return ""

    

def clean_dat (text1):

    t_out = BeautifulSoup(text1,'lxml').text

    return t_out
for item in list(dataframes.keys()):

    dataframes[item]['content']=dataframes[item]['content'].map(clean_dat)
dataframes["robotics"]['content'].iloc[2]
def removePunctuation(x):

    # Lowercasing all words

    x = x.lower()

    # Removing non ASCII chars

    x = re.sub(r'[^\x00-\x7f]',r' ',x)

    # Removing (replacing with empty spaces actually) all the punctuations

    return re.sub("["+string.punctuation+"]", " ", x)
for df in dataframes.values():

    df["title"] = df["title"].map(removePunctuation)

    df["content"] = df["content"].map(removePunctuation)

    df["content"] = df["content"].apply(lambda x:(' ').join(x.split('\n')))

    