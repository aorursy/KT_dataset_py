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
data = pd.read_csv("/kaggle/input/vgsale_1.csv")

data
before2000 = data[data["Year"]<2000]

before2000
after2000 = data.loc[data["Year"]>=2000]

after2000
dio = data[["Name", "Genre", "Global_Sales"]]

dio = data.drop_duplicates(["Name"])

dio
dio = data.Genre.value_counts().plot.bar()

dio
dio2 = data[["Genre", "Global_Sales"]]

dio2 = data.groupby("Genre")["Global_Sales"].sum()

dio2
dio1 = dio2.plot.bar()

dio1
data.groupby("Year")["Name"].nunique().plot(figsize=(18, 10))
publishers = data.groupby("Publisher")["Name"].nunique().nlargest(3)

publishers
platforms = data[data["Publisher"].isin(publishers.index)].groupby(["Publisher","Platform"])["Name"].nunique()

platforms 
platforms = platforms.unstack(level=0)

platforms

platforms.plot.bar(figsize=(22, 15), layout=(3, 2), stacked=True)
before2000 = data[data["Year"]<2000][["NA_Sales", "EU_Sales", "JP_Sales"]].sum()

before2000
before2000.plot.pie()
after2000 = data[data["Year"]>=2000][["NA_Sales", "EU_Sales", "JP_Sales"]].sum()

after2000
after2000.plot.pie()