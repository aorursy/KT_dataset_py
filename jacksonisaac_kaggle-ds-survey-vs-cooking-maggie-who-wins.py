# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
input_file_multi = '../input/multipleChoiceResponses.csv'
input_df = pd.read_csv(input_file_multi)

input_df.head()
input_df.columns
#duration_col = 'Time from Start to Finish (seconds)'
input_data = input_df.iloc[1:]
input_data.head()
input_data = input_data.rename({"Time from Start to Finish (seconds)":"Duration"}, axis=1)
input_data.head()
# input_data.columns.values[0] = "Duration"
# input_data.head()
input_data.columns
# input_data.Duration
# input_data.iloc[:, 0]
input_data.head()
input_data['Duration'] = pd.to_numeric(input_data['Duration'])
input_data.head()
duration = 5 * 60
input_5min = input_data.loc[input_data['Duration'] <= duration]

input_5min.head()
input_5min.shape
(3747/23859)*100
input_5min.groupby('Q1')['Q1'].count()
#input_5min.groupby(['Q3','Q1'])['Q1'].count().sort_values(ascending=False)
input_5min.groupby(['Q3','Q1'])['Q1'].count().reset_index(name='count') \
                             .sort_values(['count'], ascending=False)
((625+555)/3747)*100
input_india = input_5min.loc[input_5min['Q3'] == 'India']
input_india.head()
input_india.shape
input_india.groupby(['Q7', 'Q1'])['Q1'].count().reset_index(name='count') \
                              .sort_values(['count'], ascending=False)
print(230/760 * 100)
print(106/760 * 100)
input_in_male_stud = input_india.loc[(input_india['Q7'] == 'I am a student') &
                                    (input_india['Q1'] == 'Male')]

input_in_male_stud.head()
input_in_male_stud.shape
input_in_male_stud.groupby(['Q4', 'Q5'])['Q5'].count().reset_index(name='count')\
                            .sort_values(['count'], ascending=False)
print(95/230 * 100)
print((95+30)/230 * 100)
input_india.shape
input_india.groupby(['Q4', 'Q5'])['Q5'].count().reset_index(name='count')\
                            .sort_values(['count'], ascending=False)
print((243+121)/762 * 100)
