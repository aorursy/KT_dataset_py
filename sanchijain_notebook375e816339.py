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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

cities=pd.read_csv('../input/cities_r2.csv')

cities.rename(columns={'state_name':'State'},inplace=True)

data=cities.groupby('State').agg({'population_female':np.sum,'literates_female':np.sum})

data['literacy_percent_f']=data['literates_female']*100/data['population_female']

data_display=data['literacy_percent_f'].sort_values(ascending=False)

#print(data_display)

sns.barplot(x=data_display,y=data_display.index)

plt.show()








