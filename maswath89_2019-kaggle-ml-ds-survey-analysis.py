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
mcr = pd.read_csv ('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

questions = pd.read_csv ('/kaggle/input/kaggle-survey-2019/questions_only.csv')

survey = pd.read_csv ('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

other = pd.read_csv ('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
mcr.head()
import seaborn as sns
mcr.head(1)
order = mcr['Q1'].value_counts().index

sns.countplot(mcr['Q1'], order = order)