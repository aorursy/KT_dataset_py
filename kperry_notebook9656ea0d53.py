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
open_data = pd.read_csv("../input/open-data/countries.csv")

happiness = pd.read_csv("../input/world-happiness/2015.csv")



scores = pd.read_csv("../input/open-data/scores.csv")
scores.head()
od = open_data.copy().rename(index=str, columns={ "Country Name" : "Country"})

od.head()
od = open_data.copy().rename(index=str, columns= { "Country Name" : "Country"})

od.head()

happiness.info()
np.sort( od["Country"].unique() )





np.sort( happiness["Country"].unique())
np.intersect1d( od["Country"], happiness["Country"])


open_data_happiness = pd.merge(

        open_data.rename(index=str, columns={"Country Name" : "Country"}), 

        happiness, on= ["Country"], how="left"

    )



open_data_happiness["Country"].count()



import re

[ col for col in list(open_data_happiness) if re.match(".*Score$", col)]
sns.regplot(x="2015 Score", y="Happiness Score", data=open_data_happiness)