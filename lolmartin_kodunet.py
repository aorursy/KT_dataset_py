# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

%matplotlib inline

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/astronauts.csv")
df.info()
df
df[["Name", "Gender", "Space Flight (hr)"]].sort_values(["Space Flight (hr)"], ascending=[False]).reset_index(drop=True).shift()[1:11]

# df.groupby(["Birth Place", "Alma Mater", "Undergraduate Major"]).size().reset_index(name='count').sort_values('count', ascending=False).reset_index(drop=True)[:26]

# df.groupby(["Birth Place", "Space Walks"]).size().reset_index(name='count').sort_values('count', ascending=False)[:50].reset_index(drop=True)

df.groupby(["Undergraduate Major","Space Flights"]).size().reset_index(name='count').sort_values('count', ascending=False).reset_index(drop=True)[:26]

#df.groupby(["Space Flights", "Undergraduate Major"]).size().reset_index(drop=True)[:26]

#df.groupby(["Space Flights", "Undergraduate Major"], sort=False).size().reset_index(drop=True)

# df.groupby(["Undergraduate Major","Space Flights"]).count()



# mida = df.groupby(["Name", "Gender", "Death Mission"]).size()





# df.groupby(["Name", "Gender : 'Female'", "Death Mission"]).size()
# df.groupby(["Name", "Birth Place","Undergraduate Major", "Space Walks"]).size().reset_index(name='count').sort_values('count', ascending=False).reset_index(drop=True)[:51]



df.groupby(["Name", "Birth Place","Undergraduate Major", "Space Walks"], sort=False).size()[1:50]
df["Birth Place"]
df["Undergraduate Major"].unique()



round(df["""Space Flight (hr)"""].mean(), 1)
