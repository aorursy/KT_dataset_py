# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv", usecols=['Category'])

categories = df['Category'].unique()



print(f"{len(categories)} categories:")

print(categories)
frequency = df['Category'].value_counts()



# frequency is a pandas Series, so we'll transform it in a DataFrame just for presentation purposes

frequency_dist = pd.DataFrame(frequency)

frequency_dist.columns = ['Frequency']

frequency_dist.index.name = 'Category'



# Using head(10) to show only the first 10 lines

frequency_dist.head(10)
frequency_dist['Relative Frequency (%)'] = (frequency_dist['Frequency']/sum(frequency_dist['Frequency']))*100



# Using head(10) to show only the first 10 lines

frequency_dist.head(10)
import plotly.express as px



fig = px.bar(frequency)

fig.update_layout(title='Google Play Store Apps by Category',

                  xaxis_title='Category',

                  yaxis_title='# of Apps')

fig.show()