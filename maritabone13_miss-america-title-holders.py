# All my life I been interested in misses, so when I found this data about miss america titleholders i couldn't avoid working with it.

# My objective was to learn about the most common age of misses when they won the tittle and where were they usually from.



## for that, first is necessary to import the following packages.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Establishing the data set from a local repository



dat = pd.read_csv("../input/miss_america_titleholders.csv", sep = ',')
## Turning data set into a data frame



data = pd.DataFrame(dat)
## number of rows and columns un the data frame



data.shape
print(type(data))
data.head(3)
data.isnull().sum()
citie = pd.read_csv("../input/cities.csv")
cities = pd.DataFrame(citie)
cities.head(3)
#Analysis of Miss America Titleholders dataset



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["figure.figsize"]=[18.0, 10.0]

plt.style.use("ggplot")
# Fashion ages



ages = data['age'].value_counts() 



plt.figure(num = None, figsize = (8, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')

ages[:10].plot.bar()

plt.title('Most popular ages of titleholders')

plt.ylabel('Ages')      

plt.show();
# With the graphic above I learn that the most popular age when misses receive tittles is 21



staters = data['state_or_district'].value_counts() 



plt.figure(num = None, figsize = (8, 6), dpi = 80, facecolor = 'w', edgecolor = 'k')

staters[:10].plot.bar()

plt.title('Most popular districts of titleholders')

plt.ylabel('winners')      

plt.show();
# Finally with the last graphic I learned that usually misses that won tittles are from New York which for me is normal after knowing that after winning the tittle miss America have to live in Manhattan- New York. So may that may it more easy because the feel at at home.  