# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

df




pd.get_dummies(df) # Notice that we can eliminate one column of each since this information is contained in the others



pd.get_dummies(df, drop_first=True)
df.sort_values("App")



df.groupby("Category")["App"].last().to_frame()
df.sample(frac = 0.5, random_state = 2)
df.sample(frac = 0.5, random_state = 2).reset_index(drop = True) 
app = "Rating"



# allows us to iterate fast over columns

df[f'{app}'].to_frame()
# first let's use applymap to convert to standarize the text

df = df.applymap(lambda x: x.lower() if type(x) == str else x)



mapping = {"App":0, "Category":1}



print("PROBLEM: Applies to the whole df but retruns None")

df.applymap(mapping.get)



print("Get the correct result but you have to specify the colums. If you don't want to do this, check the next result")

df[["App", "Category"]].applymap(mapping.get)



print("Condtional apply map: if can map --> map else return the same value")

df = df.applymap(lambda x: mapping[x] if x in mapping.keys() else x)

df