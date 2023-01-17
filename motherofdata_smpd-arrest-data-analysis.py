# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline



figsize = (16,8)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Now that my data is pretty, I am telling Panda to read it & calling it 'arrest_data' from this point further. I don't actually know if this needs to be here or not... but I'm gonna keep it. 



arrest_data = pd.read_csv('../input/smpd-data/arrest list - total.csv')
# These are the columns (fields) available on arrest_data. An index I suppose: 



arrest_data.columns
# Compute statistics regarding the relative quanties of arrests, warnings, and citations (work in progress)



def compute_stats(data):

    n_arrests = len(data)



    return(pd.Series(data = {

        '# of arrests': n_arrests

    }))
# Attempting to calculate percent of arrests by race



def prcnt(x, y):

    if not x and not y:

       print('x = 0%\ny = 0%')

    elif x < 0 or y < 0:

       print("can't be negative!")

    else:

       total = 100 / (x + y)

       x *= total

       y *= total

       print('x = {}%\ny = {}%'.format(x, y))

        

        

arrest_data['Officer_Last_Name'].value_counts(prcnt)
#I have data for 3 full years of arrests. How many total arrests for that period were there?



arrest_data['Incident_Nr'].count()
#Naming the variable? does this need to be done?



total_arrests = arrest_data['Incident_Nr'].count()
# Now I am pulling the 'Officer' field my arrest_data & using the value.count function to see how many arrests each officer has made. 

# The square brackets call the list of officers and then counts the values. 



arrest_data['Officer'].value_counts()

# same as above but for gender



arrest_data['Gender'].value_counts()
# Create chart of arrests by gender using compute_stats



arrest_data.groupby('Gender').apply(compute_stats)
# race 



arrest_data['Race'].value_counts()
# Create chart of arrests by race using compute_stats



arrest_data.groupby('Race').apply(compute_stats)
# age



arrest_data['Age'].value_counts()
# Create chart of # of arrests by age using compute_stats (asc by age)



arrest_data.groupby('Age').apply(compute_stats)
# Not accurate yet. Do not use. 



fig, ax = plt.subplots()

ax.set_xlim(14,83)

for race in arrest_data['Race'].unique():

    s = arrest_data[arrest_data['Race'] == race]['Age']

    s.plot.kde(ax=ax, label=race)

ax.legend()
# Not accurate yet. Do not use. 



arrest_data.groupby(['Officer_Last_Name','Race']).apply(prcnt)