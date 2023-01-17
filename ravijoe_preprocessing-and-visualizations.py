# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
multiple_choice_responses
newmc=multiple_choice_responses.rename(columns=multiple_choice_responses.iloc[0]).drop(multiple_choice_responses.index[0])

# multiple_choice_responses
newmc
ax=sns.countplot(x="Select the title most similar to your current role (or most recent title if retired): - Selected Choice", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax.tick_params(axis = 'x', which = 'major', labelsize = 18)

plt.gcf().set_size_inches(18, 6)

ax=sns.countplot(x="What is the highest level of formal education that you have attained or plan to attain within the next 2 years?", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax.tick_params(axis = 'x', which = 'major', labelsize = 18)

plt.gcf().set_size_inches(18, 6)
ax=sns.countplot(x="What is your gender? - Selected Choice", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax.tick_params(axis = 'x', which = 'major', labelsize = 18)

plt.gcf().set_size_inches(18, 6)
ax=sns.countplot(x="In which country do you currently reside?", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=75)

ax.tick_params(axis = 'x', which = 'major', labelsize = 12)

plt.gcf().set_size_inches(18, 6)
ax=sns.countplot(x="What is your age (# years)?", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=75)

ax.tick_params(axis = 'x', which = 'major', labelsize = 12)

plt.gcf().set_size_inches(18, 6)
ax=sns.countplot(x="What is the size of the company where you are employed?", data=newmc)

ax.set_xticklabels(ax.get_xticklabels(),rotation=75)

ax.tick_params(axis = 'x', which = 'major', labelsize = 12)

plt.gcf().set_size_inches(18, 6)
other_text_responses
questions_only
survey_schema