# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt   

import os

import seaborn as sns  

os.getcwd()

df = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1", low_memory=False)

df.head

df.columns
#Distribution of crimes by district in descending Order

descending_order = df['DISTRICT'].value_counts().index

sns.countplot("DISTRICT", data = df, order = descending_order)

##District B2 has the highest crime rate followind by C11 and D4 in that order 

##Distribution of crimes by Offense code group top 12
##Distribution of crimes by Offense code group top 12

crimes_offense = pd.DataFrame({'Count' : df.groupby(["YEAR","OFFENSE_CODE_GROUP"]).size()}).reset_index().sort_values('Count',ascending = False).head(12)

crimes_offense

sns.barplot(x = "OFFENSE_CODE_GROUP",y= "Count",hue="YEAR", data=crimes_offense)
##Distribution of crimes by YEAR

sns.countplot("YEAR", data = df)
##Distribution of crimes by Month

sns.countplot("MONTH", data = df)

#the highest crime rates are in the months Aug, Sep, & Oct
##Distribution of crimes by Hour

sns.countplot("HOUR", data = df)

#Crime rates are at a minimum at 4 and 5 am in the morning

#Crime rates peak at between 4-6 pm
#YEARWISE breakup of Crimes by District

sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest

            hue="MONTH",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=df,

            kind="count")

#Crime rates are consistent in the districts B2, C11 & D4 across the 4 years
#YEARWISE & Day-WISE breakup of Crimes by District

sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest

            hue="DAY_OF_WEEK",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=df,

            kind="count"

            )

# Crime rates are the highest on Wed, Thu & Friin most of the districts