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

Death_Rates = pd.read_csv("../input/leading-causes-of-death-usa/Death_Rates1900-2013.csv")

leading_cause_death = pd.read_csv("../input/leading-causes-of-death-usa/leading_cause_death.csv")
Death_Rates.head()

#age-adjusted rates = "rates that would have existed if the population under study had the same age distribution as the "standard" population"

#according to Wikipedia: "Mortality rate is typically expressed in units of deaths per 1,000 individuals per year"
leading_cause_death.head()
Death_Rates.describe()
leading_cause_death.describe()
rates = Death_Rates.loc[Death_Rates['Leading Causes'] == 'Cancer']

print(rates)
#https://python-graph-gallery.com/30-basic-boxplot-with-seaborn/

#https://medium.com/@vladbezden/how-to-set-seaborn-plot-size-in-jupyter-notebook-63ffb1415431



# library & dataset

import seaborn as sns

import matplotlib.pyplot as plt

 

# Make boxplot for one group only

plt.figure(figsize=(10, 5))



sns.boxplot(x=rates['Age Adjusted Death Rate'], palette="Reds")

#Learned to change color from the following site: https://python-graph-gallery.com/33-control-colors-of-boxplot-seaborn/



plt.show()

state_where_deaths = leading_cause_death['STATE']

print(state_where_deaths.value_counts())

print(state_where_deaths.value_counts().keys())



#Data set has each variable repeat approximately the same number of times, for every variable. The same type of data is being gathered every year.
year_2002 = leading_cause_death.loc[leading_cause_death['YEAR'] == 2002][leading_cause_death['CAUSE_NAME'] == 'Influenza and pneumonia']

#limiting the year to only 2002, when I was born, and focusing on data pertaining to influenza and pneumonia that year



year_2002.head()
import numpy as np

import matplotlib.pyplot as plt



height = year_2002['DEATHS'].astype('int32')

bars = year_2002['STATE']

y_pos = np.arange(len(bars))



plt.figure(figsize=(50,40))

plt.rcParams.update({'font.size': 50})

plt.xlabel('Number of Deaths Recorded in Each State')

plt.ylabel('U.S. States')

plt.title('Overall Deaths Due to Influenza and Pneumonia in 2002 By State')



plt.bar(y_pos, height, color=['pink'], edgecolor='green', linewidth=5)



#According to the CDC, the "Flu Awareness Ribbon's colors are pink and green, chosen to represent good health and life"



# Create names on the x-axis

plt.xticks(y_pos, bars, rotation=90)

 

# Show graphic

plt.show()
year_2002.loc[year_2002['STATE']=='Alaska']
year_1999 = leading_cause_death.loc[leading_cause_death['YEAR'] == 1999][leading_cause_death['CAUSE_NAME'] == 'Diabetes']
# Import library and dataset

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



# Make default histogram

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(15,10))

#font size changed with help from this site: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

sns.distplot(year_1999['DEATHS'], color = "#0000FF")

#blue is the color of diabetes awareness

plt.xlabel('Number of Deaths Recorded')

plt.title('Overall Deaths Due to Diabetes in 1999')







plt.show()