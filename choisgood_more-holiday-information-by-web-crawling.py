# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime



df_list = []

for year in range(2011, 2017):

    df = pd.read_html("https://www.timeanddate.com/holidays/us/{}".format(year))[0]



    df.columns = df.columns.get_level_values(0)

    df.drop("Unnamed: 1_level_0", axis=1, inplace=True)



    df = df.iloc[:-1]

    df = df.dropna(how="all")



    #

    # IMPORTANT: You may need to parse date differently. (It depends on where kaggle kernel server is located)

    #

    df['Date'] = ("{} ".format(year) + df['Date']).apply(lambda x: datetime.strptime(x, "%Y %b %d"))



    df['Name'] = df['Name'].str.lower()

    df['Type'] = df['Type'].str.lower()

    df['Details'] = df['Details'].str.lower()



    df = df.drop_duplicates(['Date', 'Name', 'Type'])

    df_list.append(df)
df.head()