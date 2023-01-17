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
df = pd.read_csv("/kaggle/input/chocolate-bar-2020/chocolate.csv",index_col=0)

df
df.info()
# Top 10 rated companies

company_df = df[["company","rating"]]



company_df.groupby(['company']).mean().sort_values(by=(["rating"]),ascending=False)[:10]
# Top 10 rated company locations

rating_location_df = df[["company_location","rating"]]



rating_location_df.groupby(['company_location']).mean().sort_values(by=(["rating"]),ascending=False)[:10]
# Sugar or no sugar?

sugar_df = df[["sugar","sweetener_without_sugar","rating"]]



sugar_df.groupby(['sugar']).mean().sort_values(by=(["rating"]),ascending=False)
sugar_df.groupby(['sweetener_without_sugar']).mean().sort_values(by=(["rating"]),ascending=False)
# Salt or no salt?

sugar_df = df[["salt","rating"]]



sugar_df.groupby(['salt']).mean().sort_values(by=(["rating"]),ascending=False)
# How much cocoa?

cocoa_percent_df = df[["cocoa_percent","rating"]]



# Categorize cocoa percentage

cocoa_percent_df["cocoa_percent_range"] = pd.cut(cocoa_percent_df["cocoa_percent"],

                                                 bins=[30,42,66,77,88,100],labels=["30-41%","41-65%","66-76%",

                                                                                  "77-87%","88-100%"])



cocoa_percent_df = cocoa_percent_df.drop("cocoa_percent",axis=1)

cocoa_percent_df.groupby(["cocoa_percent_range"]).mean().sort_values(by=(["rating"]),ascending=False)
# Vanilla or no vanilla?

sugar_df = df[["vanilla","rating"]]



sugar_df.groupby(['vanilla']).mean().sort_values(by=(["rating"]),ascending=False)
# Cocoa butter or no cocoa butter?

cocoa_df = df[["cocoa_butter","rating"]]



cocoa_df.groupby(['cocoa_butter']).mean().sort_values(by=(["rating"]),ascending=False)
# More ingredients or less?

ingredients_df = df[["counts_of_ingredients","rating"]]



ingredients_df.groupby(['counts_of_ingredients']).mean().sort_values(by=(["rating"]),ascending=False)