# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import warnings
import string
import time
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15
data = pd.read_csv('../input/aac_shelter_outcomes.csv')
data.head()
data.drop_duplicates(subset='animal_id', keep='first', inplace=True)
age_upon_outcome = data['age_upon_outcome'].value_counts().head(10)
plt.figure(figsize=(12,8))
_ = sns.barplot(age_upon_outcome.index, age_upon_outcome.values)
plt.xlabel("Age Upon Outcome")
plt.ylabel("Count")
for item in _.get_xticklabels():
    item.set_rotation(30)
plt.show()
animal_type = data['animal_type'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(animal_type.values, explode=explode, labels=animal_type.index)
plt.axis('equal')
plt.tight_layout()
plt.show()
breed = data['breed'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(breed.values, explode=explode, labels=breed.index)
plt.axis('equal')
plt.tight_layout()
plt.show()
color = data['color'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(color.values, explode=explode, labels=color.index)
plt.axis('equal')
plt.tight_layout()
plt.show()
sex_upon_intake = data['sex_upon_outcome'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(sex_upon_intake.values, explode=explode, labels=sex_upon_intake.index)
plt.axis('equal')
plt.tight_layout()
plt.show()
outcome_type = data['outcome_type'].value_counts().head(4)
plt.figure(figsize=(6,6))
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05,0.05)
plt.pie(outcome_type, colors = colors, labels=outcome_type.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')  
plt.tight_layout()
plt.show()