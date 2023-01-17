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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
min_occupancy = 125

max_occupancy = 300



# Plot options

width = 0.5       # the width of the bars: can also be len(x) sequence

color_pal = ['blue', 'cyan', 'green', 'yellow', 'white', 

             'darkorange', 'salmon', 'tan', 'red', 'grey', 

             'black']
data_file = "/kaggle/input/santa-workshop-tour-2019/family_data.csv"

data = pd.read_csv(data_file)



submission_file = "/kaggle/input/santa-workshop-tour-2019/sample_submission.csv"

solution = pd.read_csv(submission_file)
data = pd.merge(solution, data, how="left", on="family_id")
data = data[['family_id', 'choice_0', 'choice_1', 'choice_2',

       'choice_3', 'choice_4', 'choice_5', 'choice_6', 'choice_7', 'choice_8',

       'choice_9', 'assigned_day', 'n_people']]
# Get assigned choice

def add_row_preference_cost(row):    

    # Look for a better choice

    for i in range(9, -1, -1):

        if row['choice_' + str(i)] == row['assigned_day']:

            return i



def get_assigned_choice(merge):

    merge['assigned_choice'] = merge.apply(add_row_preference_cost, axis=1).fillna(int(10))

    return np.array(merge.assigned_choice)



data["assigned_choice"] = get_assigned_choice(data).astype(int)
data.head()
num_families, num_days, choices = 5000, 100, 11 # 10: 11: the assigned day

data_array = np.zeros(num_families * num_days * choices

                     ).reshape(num_families, num_days, choices)

solution_array = np.zeros(num_families * num_days * choices

                         ).reshape(num_families, num_days, choices)
for f_id in data.family_id:

    for c_id in range(11):

        # Note that we decrease the day index by 1 to start at 0

        data_array[f_id, data.loc[f_id, "assigned_day"] - 1, c_id

                  ] = data.loc[f_id, "n_people"]

    

    solution_array[f_id, data.loc[f_id, "assigned_day"] - 1, data.loc[f_id, "assigned_choice"]

                  ] = data.loc[f_id, "n_people"]
# Occupancy per day

occupancy_per_day = np.sum(data_array, axis = 0)[:, 0]
# Choices per day

choices_per_day = np.sum(solution_array, axis = 0)
plt.figure(figsize=(20,10))

day = np.arange(solution_array.shape[1])



for c in range(11):

    plt.bar(day, choices_per_day[:, c], bottom=np.sum(choices_per_day[:, 0:c], axis=1), color=color_pal[c])



plt.ylabel('Number of people')

plt.axhline(y = min_occupancy, linewidth=3, color='r', ls='--')

plt.axhline(y = max_occupancy, linewidth=3, color='r', ls='--')

plt.title('Solution visualisation', size=20)



plt.show()
# Generate data for 3D plot

x,y,z = solution_array.nonzero()

solution_array_df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'n_people': data.n_people, 'assigned_choice': data.assigned_choice })
# Display plot

fig = px.scatter_3d(solution_array_df, x='x', y='y', z='z', size= 'n_people', color='assigned_choice')

fig.show()