#Importing the necessary packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing the data
athlete = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')
#Having a look at the athletes data
athlete.head()
#Having a look at the regions data
regions.head()
#In the below section, we are trying to find the variation in the weight of the athletes taking part in olympics irrespective of the sport
athlete["Weight"].plot(kind = "hist", bins = 200,figsize = (12,6), xlim = (0,150))
athlete_weight = athlete[(athlete["Weight"]>=60) &  (athlete["Weight"]<=76)]
athlete_weight["Weight"].plot(kind = "hist", bins = 16, color="Red", title = "Height >= 60 and <= 76")
plt.xlabel("Weight", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution by Weight",fontsize=16)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()
#In the below section, we are trying to find the variation in the Height of the athletes taking part in olympics irrespective of the sport
f, axes = plt.subplots(1, 2, sharex = True, figsize=(12,6))
athlete["Height"].plot(kind = "hist", bins = 100, ax = axes[0], title = "Distribution by Height")
athlete_height = athlete[(athlete["Height"]>=160) &  (athlete["Height"]<=190)]
athlete_height["Height"].plot(kind = "hist", bins = 30, color="Red", ax = axes[1], title = "Height >= 160 and <= 190")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
#Analysis by Age
athlete["Age"].plot(kind = "hist")
athlete_age = athlete[(athlete["Age"]>=20) &  (athlete["Age"]<=26)]
athlete_age["Age"].plot(kind = "hist", color="Red")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution by Age")
plt.show()
#The Relation between the athlete's height and weight is analyzed using the below joint plot
x = sns.jointplot(data = athlete, x = 'Height', y = 'Weight', kind = 'scatter', ylim = (20, 160), size = 8)
#Using Facet grid to identify the relation between the athlete's height and weight based on the seasons
g = sns.FacetGrid(athlete, row = 'Year', col ='Sex', hue="Season")
g = g.map(plt.scatter, 'Weight', 'Height')
#Using joint plots to identify the difference
x = sns.jointplot(data = athlete[athlete['Season']== "Summer"], x = 'Height', y = 'Weight', kind = 'reg', ylim = (20, 160), color="Yellow", size = 10)
y = sns.jointplot(data = athlete[athlete['Season']== "Winter"], x = 'Height', y = 'Weight', kind = 'reg', ylim = (20, 160), color="Blue", size = 10)
