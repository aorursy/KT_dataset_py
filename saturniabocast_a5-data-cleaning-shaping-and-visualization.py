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
df = pd.read_csv("/kaggle/input/Perceptual Attribute Database.csv")
df
#I don't need the ns, so I'll remove those columns 

df = df.drop(['Sound n', 'Color n', 'Manip n', 'Motion n', 'Emotion n'], axis=1)
df.head()
#Do any of the domains correlate with emotion?

import matplotlib.pyplot as plt
x=df[["Sound mean"]]
y=df[['Emotion mean']]
plt.scatter(x,y, alpha=0.5, c="purple")
x=df[["Color mean"]]
y=df[['Emotion mean']]
plt.scatter(x,y, alpha=0.5, c="purple")
x=df[["Manip mean"]]
y=df[['Emotion mean']]
plt.scatter(x,y, alpha=0.5, c="purple")
x=df[["Motion mean"]]
y=df[['Emotion mean']]
plt.scatter(x,y, alpha=0.5, c="purple")
#Visualization of each domain's ratings

hist=df.hist(figsize=(14,10))
#Importance of sound to words. Using a pie chart since comparison is only two categories.

def above_or_below(mean):
    if mean >= 3:
        return "More important"
    else:
        return "Less important"
    
    
df['Sound mean'].apply(above_or_below).value_counts().plot(kind='pie', legend=True)
#Importance of color to words

def above_or_below(mean):
    if mean >= 3:
        return "More important"
    else:
        return "Less important"
    
    
df['Color mean'].apply(above_or_below).value_counts().plot(kind='pie', legend=True)
#Importance of manipulation to words

def above_or_below(mean):
    if mean >= 3:
        return "More important"
    else:
        return "Less important"
    
    
df['Manip mean'].apply(above_or_below).value_counts().plot(kind='pie', legend=True)
#Importance of motion to words

def above_or_below(mean):
    if mean >= 3:
        return "More important"
    else:
        return "Less important"
    
    
df['Motion mean'].apply(above_or_below).value_counts().plot(kind='pie', legend=True)