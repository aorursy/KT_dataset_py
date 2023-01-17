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
import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option("display.max_columns", None)

plt.style.use("ggplot")
crime = pd.read_csv("../input/crime.csv", encoding="latin")

offense_codes = pd.read_csv("../input/offense_codes.csv", encoding="latin")
print("crime data:")

print(crime.info())

print(crime.head())

print("\n\n")



print("offense data:")

print(offense_codes.info())

print(offense_codes.head())

print("\n\n")
crime["Lat"].replace(-1, None, inplace=True)

crime["Long"].replace(-1, None, inplace=True)
crime_type = crime["OFFENSE_CODE_GROUP"].value_counts()

crime_type.plot.bar()

plt.title("Types of Crime")

plt.xlabel("Types")

plt.ylabel("# of Crimes")

plt.show()
crime_day = crime["DAY_OF_WEEK"].value_counts()

crime_day.plot.line()

plt.title("Relationship between Crime and Day")

plt.xlabel("Day of a Week")

plt.ylabel("# of Crimes")

plt.show()
crime_year = crime["YEAR"].value_counts(sort=False)

crime_year.plot.line()

plt.title("Relationship between Crime and Year")

plt.xlabel("Year")

plt.ylabel("# of Crimes")

plt.show()
shooting = crime[crime["SHOOTING"] == "Y"]

shooting_day = shooting["DAY_OF_WEEK"].value_counts()

shooting_day.plot.line()

plt.title("Relationship between Shooting and Day")

plt.xlabel("Day of a Week")

plt.ylabel("# of Shooting")

plt.show()

shooting_year = shooting["YEAR"].value_counts(sort=False)

shooting_year.plot.line()

plt.title("Relationship between Shooting and Year")

plt.xlabel("Year")

plt.ylabel("# of Shooting")

plt.show()
sns.scatterplot(x="Lat", y="Long", alpha=0.01, data=crime)

plt.show()



crime_district = crime["DISTRICT"].value_counts()

crime_district.plot.bar()

plt.title("Crime in Different Districts")

plt.xlabel("Districts")

plt.ylabel("# of Crimes")

plt.show()
sns.scatterplot(x="Lat", y="Long", alpha=0.2, data=shooting)

plt.show()