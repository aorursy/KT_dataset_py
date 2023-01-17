# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/vgsales.csv")

df

myydud = ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales"] #https://stackoverflow.com/questions/27018622/pandas-groupby-sort-descending-order

grupeeritud = vgs.groupby("Publisher")[myydud].sum().reset_index()

järjestatudGRP = grupeeritud.sort_values(myydud, ascending=False)

järjestatudGRP
järjestatudGRP.head(15) #näitab 15 enim müünud publisheri
#See plokk ei tööta

#df.plot.scatter("Year","Global_Sales",)

#df = pd.DataFrame(järjestatudGRP["Global_Sales"], columns=["Year"])

#s = järjestatudGRP["Global_Sales"]

#df = pd.DataFrame(s, columns=["Year"])

#plot.scatter(s, "Year")

#vark = df.groupby("Year")[järjestatudGRP["Global_Sales"].reset_index()


g = sns.FacetGrid(df, hue="Global_Sales", size=5)

g.map(plt.scatter, "Year", "Global_Sales", s=50, alpha=.7)

#Siin on näha scatterploti sellest, kuhu enamus mängud oma müügiarvu poolest paigutuvad. Enamik

#mänge müüb vähem kui miljon, ning ainult mõningad on müünud üle 30 miljoni koopia.