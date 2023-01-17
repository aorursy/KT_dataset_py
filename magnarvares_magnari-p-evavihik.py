# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")
df
kuupäev = pd.to_datetime(df["Date"])

kuupäev.dt.year.value_counts()
surnud= df["Ground"]+df["Fatalities"]



a = df.groupby(pd.to_datetime(df.Date).dt.year)["Fatalities"].sum()

a.plot()
elus_protsent=(df["Aboard"]-df["Fatalities"])/df["Aboard"]*100

df = df.assign(Elus_Protsent = elus_protsent)

b=df.groupby(pd.to_datetime(df.Date).dt.year)["Elus_Protsent"].mean()

b.plot()
op = df["Operator"].value_counts().head(20)

op.plot.barh()
df["Type"].value_counts().head(10).plot.barh()