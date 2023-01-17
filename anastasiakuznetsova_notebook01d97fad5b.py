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



import operator
df = pd.read_json(open("../input/train.json", "r"))
df.head()


df["words"] = df["description"].apply(lambda x: (x.split(" ")))
df["words"].head()
words = {}

for i in df["words"]:

    for word in i:

        tmp = word.lower()

        if tmp in words:

            words[tmp] =  words[tmp] + 1

        else:

            words[tmp] = 1

        

    
sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
sorted_words
df["features_split"] = df["features"].apply(lambda x: (x.split(", ")))
features = {}

for i in df["features"]:

    for feature in i:

        tmp = feature.lower()

        if tmp in features:

            features[tmp] =  features[tmp] + 1

        else:

            features[tmp] = 1

            

sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
sorted_features 