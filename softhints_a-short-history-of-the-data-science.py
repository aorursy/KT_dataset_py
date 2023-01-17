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
# read data

df_multi_answers = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
# helper method for visualization of value counts

import seaborn as sns

import matplotlib.pyplot as plt



def plot_value_counts(res, size, title, x, y):

    top_res = res.head(size)



    chart = sns.barplot(top_res.index, top_res.values, alpha=0.8)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.title(title)

    plt.ylabel(x, fontsize=12)

    plt.xlabel(y, fontsize=12)

    plt.show()
countries = df_multi_answers.Q3.value_counts()

plot_value_counts(countries, 15, "Top countries Kaggle", 'Scientists', 'Country')
top_countries = df_multi_answers.Q3.value_counts().head(15).index

print(top_countries)

df_multi_answers[df_multi_answers.Q3.isin(top_countries)].groupby(['Q3']).Q1.value_counts()

df_top_countries = df_multi_answers[df_multi_answers.Q3.isin(top_countries)]





pd.crosstab(df_top_countries.Q3, df_top_countries.Q1,

                  rownames=['country'], colnames=['age'])
# read data

df_other = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')

df_other.shape
primary_tool = df_other['Q14_Part_3_TEXT'].value_counts()

plot_value_counts(primary_tool, 10, 'What is the primary tool that you use at work?', 'Number of Answers', 'Tool')
import difflib 



correct_values = {}

words = df_other.Q14_Part_3_TEXT.value_counts(ascending=True).index



for keyword in words:

    similar = difflib.get_close_matches(keyword, words, n=20, cutoff=0.6)

    for x in similar:

        correct_values[x] = keyword

             

df_other["corr"] = df_other["Q14_Part_3_TEXT"].map(correct_values)
# Using similar values

correction = df_other["corr"].value_counts()

plot_value_counts(correction, 10, 'What is the primary tool that you use at work?', 'Number of Answers', 'Tool')
df_other.columns
# View sample data

df_other.head()
# list questions and column names

import pandas as pd

pd.set_option('display.max_colwidth', 1000)

df_other.head(1).T.head()
# List top values per question

for col in df_other.columns:

    print(col, end=' - ')

    print(df_other[col][0])

    display(df_other[col].value_counts().head(10))