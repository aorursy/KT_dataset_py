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
import pandas as pd

import numpy as np



import fuzzywuzzy

from fuzzywuzzy import process

import chardet



# set seed for reproducibility

np.random.seed(0)
pak=pd.read_csv("/kaggle/input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv")
f=open("/kaggle/input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv",'rb')

res=chardet.detect(f.read(100000))

res
pak=pd.read_csv("/kaggle/input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv",encoding='Windows-1252')

pak.head()
city=pak["City"].unique()

city.sort()

city
pak['City']=pak['City'].str.title()

pak['City']=pak['City'].str.strip()
pak['City'].head()
a=(pak['Province'].unique())

print(a)

type(a)

np.sort(a)
pak['Province']=pak['Province'].str.title()

pak['Province']=pak['Province'].str.strip()
pak['Province'].head()
# get the top 10 closest matches to "d.i khan"

matches = fuzzywuzzy.process.extract("d.i khan", city, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

matches
# function to replace rows in the provided column of the provided dataframe

# that match the provided string above the provided ratio with the provided string

def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):

    # get a list of unique strings

    strings = df[column].unique()

    

    # get the top 10 closest matches to our input string

    matches = fuzzywuzzy.process.extract(string_to_match, strings, 

                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)



    # only get matches with a ratio > 90

    close_matches = [i[0] for i in matches if i[1] >= min_ratio]



    # get the rows of all the close matches in our dataframe

    rows_with_matches = df[column].isin(close_matches)



    # replace all rows with close matches with the input matches 

    df.loc[rows_with_matches, column] = string_to_match

    

    # let us know the function's done

    print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"

replace_matches_in_column(df=pak, column='City', string_to_match="d.i khan")
cities = pak['City'].unique()

a=np.sort(cities)

a
# get the top 10 closest matches to "Kuram Agency"

matches = fuzzywuzzy.process.extract("Kuram Agency", city, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

matches
# use the function we just wrote to replace close matches to "Kuram Agency" 

replace_matches_in_column(df=pak, column='City', string_to_match="Kuram Agency")
cities = pak['City'].unique()

a=np.sort(cities)

a