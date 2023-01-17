# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_set = pd.read_csv("../input/menu.csv")

data_set
data_set.plot.scatter(x = 'Calories', y = 'Total Fat')
bs = {'Calories': data_set['Calories'], 'Total Fat': data_set['Total Fat']}

bs
Calories_TotalFat_data_frame = pd.DataFrame(bs)
Calories_TotalFat_data_frame.corr('pearson')
Protein = {'Protein': data_set['Protein']}
protein_data_frame = pd.DataFrame(Protein)

protein_data_frame
y = protein_data_frame['Protein'].value_counts()

y
data_set.plot.scatter(x = 'Sodium', y = 'Cholesterol')
sc = {'Sodium': data_set['Sodium'], 'Cholesterol': data_set['Cholesterol']}

sc
sodium_cholesterol_data_frame = pd.DataFrame(sc)
sodium_cholesterol_data_frame.corr('pearson')