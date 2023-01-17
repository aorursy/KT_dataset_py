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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv("../input/golden-globe-awards/golden_globe_awards.csv")
data.head()
data.isnull().sum()
data['win'].value_counts()
data["nominee"].value_counts().head(10).plot.bar()
data['category'].value_counts().head(20)
plt.figure(figsize=[15,15])

data["category"].value_counts().head(45).plot.bar()
group =data.groupby("year_award")

group.first()
group2=data.groupby(["win","category"])
group2.first()
tag="Best Performance by an Actress in a Supporting Role in any Motion Picture"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag="Meryl Streep"

data['relevent']=data["nominee"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]]
tag="Best Director - Motion Picture"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag="Best Performance by an Actor in a Supporting Role in any Motion Picture"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)

tag="Best Motion Picture - Drama"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag="Best Performance by an Actor in a Motion Picture - Drama"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag="Best Performance by an Actor in a Motion Picture - Musical or Comedy"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag = "Best Performance by an Actress in a Motion Picture - Drama"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data["relevent"]==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag="Best Original Score - Motion Picture"

data['relevent']=data["category"].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data['relevent']==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)
tag = "Best Motion Picture - Musical or Comedy"

data['relevent'] = data['category'].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data['relevent']==1]

small[small['win']==True][["year_award","nominee","win"]].tail(20)
tag = "Best Original Song - Motion Picture"

data['relevent'] = data['category'].fillna("").apply(lambda x:1 if tag.lower() in x.lower() else 0)

small = data[data['relevent']==1]

small[small['win']==True][["year_award","nominee","film","win"]].tail(20)