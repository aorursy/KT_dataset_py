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
# import libraries

import pandas as pd

import numpy as np

import os
csv_file = '/kaggle/input/plagiarismdata12/data/file_information.csv'

plagiarism_df = pd.read_csv(csv_file)



# print out the first few rows of data info

plagiarism_df.head(10)
plagiarism_df.shape
# print out some stats about the data

print('Number of files: ', plagiarism_df.shape[0])  # .shape[0] gives the rows 

# .unique() gives unique items in a specified column

print('Number of unique tasks/question types (A-E): ', (len(plagiarism_df['Task'].unique())))

print('Unique plagiarism categories: ', (plagiarism_df['Category'].unique()))
# Show counts by different tasks and amounts of plagiarism



# group and count by task

counts_per_task=plagiarism_df.groupby(['Task']).size().reset_index(name="Counts")

print("\nTask:")

display(counts_per_task)



# group by plagiarism level

counts_per_category=plagiarism_df.groupby(['Category']).size().reset_index(name="Counts")

print("\nPlagiarism Levels:")

display(counts_per_category)



# group by task AND plagiarism level

counts_task_and_plagiarism=plagiarism_df.groupby(['Task', 'Category']).size().reset_index(name="Counts")

print("\nTask & Plagiarism Level Combos :")

display(counts_task_and_plagiarism)
import matplotlib.pyplot as plt

%matplotlib inline



# counts

group = ['Task', 'Category']

counts = plagiarism_df.groupby(group).size().reset_index(name="Counts")



plt.figure(figsize=(8,5))

plt.bar(range(len(counts)), counts['Counts'], color = 'blue')