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
import datetime as dt 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/austin-animal-center-shelter-outcomes-and/aac_shelter_outcomes.csv')

print(df.shape)

df.head()



#This is to see an overview of how the data is presented
df["animal_type"] = "dog"

print(df["animal_type"])





sns.countplot(df.animal_type, palette='pastel')



#I have decided to focus soley on dogs for this assignment. 

#So I redefined the dataset to only include those animals whose animal_type is listed as dog

#I printed out a graph to check

plt.figure(figsize=(20, 8))

sns.countplot(df.sex_upon_outcome, palette='pastel')



#This graph represented the sex of the animal at the end of it's journey at the shelter

#I thought it would be interesting to known more about the dog demographics

#I am using a bar graph to show the sex vs count

#I made the graph bigger so it was easier to read
#In order to find out more about how age impacts the outcome of the animal, I need to write a function to transform the age_upon_outcome column into an integer. 





def years_old(x):  #function years_old to hold transformation

    x = str(x)   #converts all values to strings

    if x == 'nan': #if the value is null, removes

        return 0

    HowOld = int(x.split()[0])  #splits the string and converts all numbers to integers with base 0

    if x.find('year') != -1:    #if year in string, returns that numbers

        return HowOld   

    if x.find('month')!= -1:       #if month in string, returns that number divided by 12

        return HowOld  / 12

    if x.find('week')!= -1:        #if week in string, returns that number divided by 52

        return HowOld  / 52

    if x.find('day')!= -1:         #if day in string, returns that number divided by 365

        return HowOld  / 365

    else: 

        return 0                   #if no value in string, returns 0

df['AnimalAge'] = df.age_upon_outcome.apply(years_old)    #uses variable AnimalAge to apply function to dataset. 

print(df['AnimalAge'].head(5))   #print dataset head/tail to see if it worked

df['AnimalAge'].tail(5)
plt.figure(figsize=(20, 8))

sns.boxplot(x='outcome_type', y='AnimalAge', data=df, palette='pastel' )



#this plot uses the AnimalAge variable we defined above. 

#this shows the outcome of the animal along with it's relation to the animal's age
