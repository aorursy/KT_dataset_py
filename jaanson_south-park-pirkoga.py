# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

pd.set_option('display.max_rows', 20)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv("../input/All-seasons.csv")

df



# Any results you write to the current directory are saved as output.
df= df.replace('\n', '', regex=True)

df
df.info()
a=df['Character'].value_counts().head(6)

print(a)
#df.Episode.plot.hist();

#df['Character']

#print('Stan räägib', df['Character'].count()) 

#teeme uue veeru

#nr = df.Character.value_counts()

#nr3 = (nr > 3)

(pd.DataFrame({"1.tegelane" : df["Character"], 

              "3.ütlemine" : df['Line'], "4.ütluse pikkus": df["Line"].str.len()}) 

.sort_values("4.ütluse pikkus", ascending=False))

#df[['Character','Line']]
df.groupby("Episode")["Line"].str.len()
df.plot.scatter("Season", "Episode", alpha=0.2);
#df['Season'] = df['Season'].convert_objects(convert_numeric=True)

df.Episode = df.Episode.astype(float)
df["Line"].value_counts().sort_values(ascending=False)



print((df['Line'].str.contains("You bastards")==True).value_counts())

print((df['Line'].str.contains("killed Kenny")==True).value_counts())

print((df['Line'].str.contains("Oh my God, they killed Kenny!")==True).value_counts())

print(((df['Line'].str.contains("killed")==True)&(df['Line'].str.contains("Kenny")==True)&(df['Line'].str.contains("You bastards"))).value_counts())
df[['Character',Line.str.len().sort_values(ascending=False)]]
(df[["Season", "Episode", "Line"]]

 ["Line"].str.len().sort_values(ascending=False))
(df[["Character","Line"]]

 .sort_values("Line", ascending=False))
df.Line.str.slipt(', ')