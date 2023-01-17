# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from statistics import mode

import statistics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
data.shape
data.columns
data["Last Updated"].head()
data.info()
data.isnull().sum()
def getcategory(dataset):

    missing_ratings = data.loc[data["Rating"].isnull(),"Category"].unique()

    for category_name in missing_ratings:   

        nonullcategory = dataset.loc[(dataset["Category"] == category_name) & (dataset["Rating"].notnull()), ["Category","Rating"]]

        dataset.loc[(dataset["Category"] == category_name) & (dataset["Rating"].isnull()),"Rating"] = nonullcategory["Rating"].mean()

    return dataset
ratings_filled = getcategory(data)

ratings_filled.isnull().sum()
ratings_filled.loc[ratings_filled["Type"].isnull() , "Type"] = ratings_filled["Type"].mode().values



ratings_filled.isnull().sum()
null = ratings_filled.loc[ratings_filled['Content Rating'].isnull() ,:]

print(null)

replacable_value = {"Rating" : null['Category'].values ,"Reviews" : null['Rating'].values , "Size" : null['Reviews'].values,

                    "Installs" : null['Size'].values , "Type": null['Installs'].values, "Price" : null['Type'].values, 

                    "Content Rating" : null['Price'].values}


for item in replacable_value:

    ratings_filled.loc[ratings_filled['Content Rating'].isnull(),item] = replacable_value[item]



ratings_filled.loc[ratings_filled['Category'] == '1.9' , ["Category" , "Genres"]] = "Lifestyle"
replaced = {"Last Updated": null['Genres'].values , "Current Ver": null["Last Updated"].values, "Android Ver": null["Current Ver"].values}
for items in replaced:

    ratings_filled.loc[ratings_filled['Rating']=='1.9',items] = replaced[items]

ratings_filled.isnull().sum()
ratings_filled.loc[ratings_filled['Android Ver'].isnull(),:]
ratings_filled.loc[ratings_filled['Android Ver'].isnull(),'Android Ver']

model = ratings_filled.loc[ratings_filled['Category'] == "PERSONALIZATION",'Android Ver']

model.mode().values
ratings_filled.loc[ratings_filled['Android Ver'].isnull(),'Android Ver'] = model.mode().values
ratings_filled.loc[ratings_filled['Android Ver'].isnull(),:]
ratings_filled.iloc[4453]
ratings_filled.loc[ratings_filled['Current Ver'].isnull(),:]
ratings_filled["Current Ver"].value_counts()
ratings_filled.loc[ratings_filled['Current Ver'].isnull(),"Current Ver"] = ratings_filled["Current Ver"].mode().values
ratings_filled.isnull().sum()