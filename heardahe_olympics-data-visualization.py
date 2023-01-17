# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/athlete_events.csv")
df.info()
df.head(10)
pd.DataFrame(df.Name.value_counts()).head(3)
def player_summary(name):

    year_joined = df[df.Name == name].Year.unique()

    min_age = int(df[df.Name == name].Age.min())

    max_age = df[df.Name == name].Age.max()

    medals = sum(df[df.Name == name].Medal.value_counts())

    nation = df[df.Name == name].Team.unique()

    if 'M' == df[df.Name == name].Sex.unique():

        gender = 'He'

    else:

        gender = 'She'

    

    

    print('{0} joined Olympics in {1} for the first time when {4} was {2} years old. {5} earned {3} medals in total. {5} represented {6} for {7} times.'

          .format(name,year_joined.min(),min_age,medals,gender.lower(),gender,nation[0],len(year_joined)))
player_summary('Robert Tait McKenzie')
player_summary('Christine Jacoba Aaftink')
sns.countplot(df.Sex, label = 'Count')

M,F = df.Sex.value_counts()

print('Since {0} to {1}, {2} female and {3} male atheletes competed in Olympics.'.format(df.Year.min(),df.Year.max(),F,M))



Year_Sex = pd.DataFrame(df.groupby('Year')['Sex'].value_counts().unstack(fill_value=0))

Year_Sex = Year_Sex.reset_index()



Year_Sex.plot(kind = "scatter", x = 'Year', y = 'F')

Year_Sex.plot(kind = "scatter", x = 'Year', y = 'M')
Year_NOC = pd.DataFrame(df.groupby('Year')['NOC'].value_counts().unstack(fill_value=0))

Year_NOC



f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(Year_NOC,annot=False,fmt= '.1f',ax=ax)
Year_Height = pd.DataFrame(df.groupby(['Year','Sex'])['Height'].mean().unstack(fill_value=0))

Year_Height = Year_Height.reset_index()



Year_Weight = pd.DataFrame(df.groupby(['Year','Sex'])['Weight'].mean().unstack(fill_value=0))

Year_Weight = Year_Weight.reset_index()
sns.lineplot(x="Year", y="M", data=Year_Height).set_title('Average height of male atheletes')
sns.lineplot(x="Year", y="F", data=Year_Height).set_title('Average height of female atheletes')
sns.lineplot(x="Year", y="M", data=Year_Weight).set_title('Average weight of male atheletes')
sns.lineplot(x="Year", y="F", data=Year_Weight).set_title('Average weight of female atheletes')