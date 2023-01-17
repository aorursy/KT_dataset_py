# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

crime_and_incarceration_by_state = pd.read_csv("../input/prisoners-and-crime-in-united-states/crime_and_incarceration_by_state.csv")

prison_custody_by_state = pd.read_csv("../input/prisoners-and-crime-in-united-states/prison_custody_by_state.csv")

ucr_by_state = pd.read_csv("../input/prisoners-and-crime-in-united-states/ucr_by_state.csv")
import matplotlib.pyplot as plt

crime_and_incarceration_by_state
plt.hist(crime_and_incarceration_by_state.tail(50)["violent_crime_total"], edgecolor="#802400", linewidth= 3 )



plt.xlabel("# of Violent Crimes")



plt.ylabel("# of States")



plt.title("Violent Crimes Across 50 States in 2016")



font = {'size': 100}
# Description of Histogram:

# The histogram above has a center at approximately 30,000 violent crimes. The spread of the data is from 175,000 to 0 with a range of

# 175,000 violent crimes. The shape of the histogram is a major skew right, decreasing with the number of states in the data set.

# What's unusual is that there is a large gap from approximately 120,000 to approximately 160,000 violent crimes. For context, the histogram

# depicts the total number of violent crimes across 50 states in 2016.

crime_and_incarceration_by_state.tail(51)
import seaborn as sns

sns.boxplot(x="prisoner_count", data=crime_and_incarceration_by_state.tail(50), color="green" )

plt.xlabel("# of Prisoners")

plt.title("Amount of Prisoners per State")

# Description of Boxplot:

# The center of this boxplot above is approximately 150,000 prisoners. The spread is approximately 150,000 to 0 with a range of 150,000.

# What's unusual is the 3 outliars all the way in left field outside of the fence, quite goobers those states are. The boxplot depicts

# the amount of prisoners incarcerated across all 50 states.
import matplotlib.pyplot as plt

%matplotlib inline

labels = 'California', 'Texas', 'Florida', 'Illinois', 'Ohio', 'North Dakota', 'Others'

sizes = [1930, 1478, 1111, 1061, 683, 16, 10971]

explode = (0, 0, 0, 0, 0, 0, 0.1)  # only "explode" the last slice (i.e. 'Others')



fig1, ax1 = plt.subplots(figsize=(15,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90)

ax1.axis('equal')

plt.title("Murders and Manslaughter per State")
# Description of Pie Chart:

# While the center was not narrowed down specifically, I chose 5 of the top 50 states with the highest murders and manslaughters in 2016

# (California, Texas, Florida, Illinois, and Ohio). The others section maximizes the spread at 63.6% of the total murders and manslaughters

# while North Dakota has the lowest (16 the whole year) with 0.1%. The shape is of course a pie chart but the individual state slice is about

# one third of the entire distribution. What's unusual is North Dakota's shockingly low statistic but I speculate due to it's spacious land,

# it is either relatively peaceful or they have not located all of the bodies from that year. The pie chart depicts that the top 5 states 

# make up about 33% of all data.
plt.bar(labels, sizes)

labels= "2016", "2001"

sizes= [ 188311, 149852]

# Description for Bar Graph

# The bar graph depicts the amount of federal prisoners in 2016 in comparison to the amount of federal prisoners in 2001. Notice an increase

# of about 30,000 prisoners.
plt.hist(crime_and_incarceration_by_state.head(50)["vehicle_theft"], alpha=0.5, label="2001")

plt.hist(crime_and_incarceration_by_state.tail(50)["vehicle_theft"], alpha=0.5, label="2016")

plt.legend()

# Comparison

# In comparison, these two overlapped histograms have a lot in common. Firstly, they both generally have similar centers and spreads of

# data. This time it is car thefts and you can see where the orange and the blue do share trends. Granted, 2016 has more vehicle thefts

# than 2001 but nonetheless they both follow the same shape with the normal unusual outliars. Skewed right just how we like them.