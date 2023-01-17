import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/train.csv")

df
df.shape
# how representative is this sample?

df.Survived.value_counts()
# seems to be representative ...

342 / 891
# this shows that 'Sex' is categorical

df.Sex.value_counts()
df.Sex.value_counts().plot(kind='bar')
df[df.Sex=='female']
# we can see we have no nulls ... which is good

df[df.Sex.isnull()]
# how much people paid for their tickets ...

df.Fare.value_counts()
df.describe()
df.Fare.hist()
# looking good ...

df[df.Fare.isnull()]
df[df.Fare==0]
df.Ticket.describe()
df[df.Ticket=='CA. 2343']
df[df.Cabin.isnull()]
# women & children first?

df[df.Sex=='male'].Survived.value_counts()
import matplotlib.pyplot as plt



fig, axs = plt.subplots(1, 2)



df[df.Sex=='male'].Survived.value_counts().plot(kind='barh', ax=axs[0], title="Males Survived")

df[df.Sex=='female'].Survived.value_counts().plot(kind='barh', ax=axs[1], title="Females Survived")
df[df.Age<15].Survived.value_counts().plot(kind='barh', title="Child Survivors (< 15 yrs)")
fig, axs = plt.subplots(1, 2)



df[(df.Age<15) & (df.Sex=='male')].Survived.value_counts().plot(kind='barh', ax=axs[0], title="Boy Survivors")

df[(df.Age<15) & (df.Sex=='female')].Survived.value_counts().plot(kind='barh', ax=axs[1], title="Girl Survivors")