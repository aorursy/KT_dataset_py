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
df = pd.read_csv('/kaggle/input/top-datasets-kagglers-ranking/top20KagglersDatasets.csv')

df.head(7)
import plotly.express as px

countryCounts = df['country'].value_counts()

countryCounts
fig1bar = px.bar(

    x=countryCounts.index, 

    y=countryCounts, 

    labels={'x':'Country', 'y':'Count'}, 

    title='Country count'

)



fig1bar.show()
fig1pie = px.pie(

    names=countryCounts.index, 

    values=countryCounts, 

    labels={'x':'Country', 'y':'Count'}, 

    title='Country count'

)





fig1pie.show()
fig1scatterGeo = px.scatter_geo(

    df, 

    locations=countryCounts.index, 

    locationmode ='country names', 

    size=countryCounts,

    projection="natural earth"

)



fig1scatterGeo.show()
tierCounts = df['tier'].value_counts()

tierCounts
fig2bar = px.bar(

    x=tierCounts.index, 

    y=tierCounts, 

    labels={'x':'Tier', 'y':'Count'}, 

    title='Tier count'

)



fig2bar.show()
fig2pie = px.pie(

    names=tierCounts.index, 

    values=tierCounts, 

    labels={'x':'Tier', 'y':'Count'}, 

    title='Tier count'

)



fig2pie.show()
occupationCounts = df['occupation'].value_counts()

occupationCounts
fig3pie = px.pie(

    names=occupationCounts.index, 

    values=occupationCounts, 

    labels={'x':'Occupation', 'y':'Count'}, 

    title='Occupation count'

)



fig3pie.show()
organizationCounts = df['organization'].value_counts()

organizationCounts
import fuzzywuzzy

from fuzzywuzzy import process

matches = fuzzywuzzy.process.extract("nvidia", organizationCounts.index, limit=20, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

matches
# function to replace rows in the provided column of the provided dataframe

# that match the provided string above the provided ratio with the provided string

def replace_matches_in_column(df, column, string_to_match, min_ratio = 47):

    # get a list of unique strings

    strings = df[column].unique()

    

    # get the top 10 closest matches to our input string

    matches = fuzzywuzzy.process.extract(string_to_match, strings, 

                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)



    # only get matches with a ratio > 90

    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]



    # get the rows of all the close matches in our dataframe

    rows_with_matches = df[column].isin(close_matches)



    # replace all rows with close matches with the input matches 

    df.loc[rows_with_matches, column] = string_to_match

    

    # let us know the function's done

    print("All done!")
replace_matches_in_column(df=df, column='organization', string_to_match="Nvidia")
organizationCounts = df['organization'].value_counts()

organizationCounts
fig4pie = px.pie(

    names=organizationCounts.index, 

    values=organizationCounts, 

    labels={'x':'Organization', 'y':'Count'}, 

    title='Organization count'

)





fig4pie.show()
fig5wideFormatBar = px.bar(

    df, 

    y='displayName', 

    x=['totalGoldMedals', 'totalSilverMedals', 'totalBronzeMedals']

)



fig5wideFormatBar.show()
df['numericalTier'] = df['tier']

df.head()
cleanup_nums = {'numericalTier': {'grandmaster':5, 'master':4, 'expert':3}}
df.replace(cleanup_nums, inplace=True)

df.head(7)
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(14,12))

correlation = df.corr()

sns.heatmap(correlation, linewidth=0.5, cmap='Blues', annot=True)