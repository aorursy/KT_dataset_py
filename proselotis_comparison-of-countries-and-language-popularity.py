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

%matplotlib inline

import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/data-science-buzz-words-frequency/merged.csv")
data.head()
[x for x in data.columns]
fig, ax = plt.subplots()



plt.plot(data['Date'],data['Boosting_India'])

plt.plot(data['Date'],data['Boosting_UnitedStates'])

plt.plot(data['Date'],data['Boosting_Worldwide'])

plt.legend()

plt.xlabel("Date (2004 - 2020)")

plt.ylabel("Standardized Frequency")

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,

                 box.width, box.height * 0.9])



# Put a legend below current axis

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.xticks([])

plt.show()
fig, ax = plt.subplots()



plt.plot(data['Date'],data['PredictiveAnalytics_India'])

plt.plot(data['Date'],data['PredictiveAnalytics_UnitedStates'])

plt.plot(data['Date'],data['PredictiveAnalytics_Worldwide'])

plt.legend()

plt.xlabel("Date (2004 - 2020)")

plt.ylabel("Standardized Frequency")

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,

                 box.width, box.height * 0.9])



# Put a legend below current axis

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.xticks([])

plt.show()
fig, ax = plt.subplots()



plt.plot(data['Date'],data['Java_Worldwide'])

plt.plot(data['Date'],data['C++_Worldwide'])

plt.plot(data['Date'],data['Python_Worldwide'])

plt.plot(data['Date'],data['RProgramming_Worldwide'])

plt.plot(data['Date'],data['Scala_Worldwide'])

plt.plot(data['Date'],data['Sas_Worldwide'])







box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 1.1,

                 box.width, box.height * 1.1])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.xlabel("Date (2004 - 2020)")

plt.ylabel("Standardized Frequency")

plt.xticks([])

plt.show()