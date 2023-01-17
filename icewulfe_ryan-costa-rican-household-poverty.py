## This Python 3 environment comes with many helpful analytics libraries installed

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


#Import Modules



import numpy as np

import pandas as pd





# Import libraries

import seaborn as sns

import matplotlib.pyplot as plt



#Set Seaborn Style

sns.set(style="whitegrid")

# Bring training and test data into environment

df_train = pd.read_csv("../input/proxymeanstest-cr/train.csv")

df_test = pd.read_csv("../input/proxymeanstest-cr/test.csv")
# Let's look at the shape of our dataframes to confirm the number of columns and rows

print(df_train.shape)

print(df_test.shape)



print("----------------------------------------")

# Fidning info of df_train data

df_train.info()



print("----------------------------------------")

# Finding info of df_test data

df_test.info()



#We see there are 143 column entries for the train set, and 142 column entries for the test set.
# Doing a Sum Statistics to look at both df_train and df_test

df_train.describe()
df_test.describe()
#Take a look to see which columns have null values

df_train.isnull().sum()



# v2a1 is the monthly rent payment and it has the most missing values.
#Same thing for test csv (null values)

df_test.isnull().sum()



#This is the same case for the test data.


# Custom colors 

# https://s3.amazonaws.com/assets.datacamp.com/production/course_15192/slides/chapter4.pdf

# https://seaborn.pydata.org/generated/seaborn.countplot.html

custom_palette = ['#FBB4AE','#FED9A6','#B3CDE3','#CCEBC5']

target_count = sns.countplot(x="Target", data = df_train, palette= custom_palette)

#To get a better understanding of what im looking at I want to take a look at the unique values (See how many booleans there are).



df_train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'green', figsize = (6.5, 4.5), edgecolor = 'black', linewidth = 3);

plt.xlabel('Unq Values', fontsize=16)

plt.ylabel('Count', fontsize=16)
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('seaborn')





# Custom Colors

color_level_map = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_level_map = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



# Enumerate to keep a count of the iterations

# https://www.geeksforgeeks.org/enumerate-in-python/

for i, col in enumerate(df_train.select_dtypes('float')):

    # ax meaning explained (https://railsware.com/blog/python-for-machine-learning-pandas-axis-explained/)

    ax = plt.subplot(4, 2, i + 1)

    

    # Iterate through the poverty levels

    for poverty_level, color in color_level_map.items():

        

        # Plot each poverty level as a separate line, density estimate

        #dropna removes rows where value is missing

        sns.kdeplot(df_train.loc[df_train['Target'] == poverty_level, col].dropna(), ax = ax, color = color, label = poverty_level_map[poverty_level])

        #https://seaborn.pydata.org/generated/seaborn.kdeplot

        

    plt.title(f'{col.capitalize()} Density Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')

    

#Changes spacing between sublot link

plt.subplots_adjust(top = 2)
print("v2a1, Monthly rent payment")

print("v18q1, number of tablets household owns")

print("rez_esc, Years behind in school")

print("meaneduc, average years of education for adults (18+)")

print("overcrowding, # persons per room")

print("SQBovercrowding, overcrowding squared")

print("SQBdependency, dependency squared")

print("SQBmeaned, square of the mean years of education of adults (>=18) in the household")


