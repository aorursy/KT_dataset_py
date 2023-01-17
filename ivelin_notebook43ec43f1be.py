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
%%time

def load_data():

    import pandas as pd

    dfs = {"biology": pd.read_csv("../input/biology.csv"),

           "cooking": pd.read_csv("../input/cooking.csv"),

           "crypto": pd.read_csv("../input/crypto.csv"),

           "diy": pd.read_csv("../input/diy.csv"),       

           "robotics": pd.read_csv("../input/robotics.csv"),

           "travel": pd.read_csv("../input/travel.csv")}

    return dfs

    

    

df = load_data()

print(df["biology"].iloc[1])
%%time

def preprocess(x):

    from bs4 import BeautifulSoup

    import re

    import string



    # remove html tags from contents

    uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

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





# This could take a while

for d in df.values():

    d["content"] = d["content"].map(preprocess)

    

print(df["biology"].iloc[1])