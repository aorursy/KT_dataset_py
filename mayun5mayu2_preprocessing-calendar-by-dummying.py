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
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
calendar.head()
calendar.isnull().sum()
calendar.describe()
calendar["event_name_1"].unique()
len(calendar["event_name_1"].unique())
calendar["event_type_1"].unique()
calendar["event_name_2"].unique()
calendar["event_type_2"].unique()
event_name = set(calendar["event_name_1"].unique()) & set(calendar["event_name_2"].unique())

print(event_name)
event_type = set(calendar["event_type_1"].unique()) & set(calendar["event_type_2"].unique())

print(event_type)
event_name_one = calendar["event_name_1"]

event_name_one = pd.get_dummies(event_name_one)

event_name_one.head()
event_name_two = calendar["event_name_2"]

event_name_two = pd.get_dummies(event_name_two)

event_name_two.head()
event_names = pd.merge(event_name_one, event_name_two, right_index=True, left_index=True)

event_names.head()
event_names.columns
event_names['Easter'] = 0

event_names['Cinco De Mayo'] = 0

event_names['OrthodoxEaster'] = 0

event_names["Father's day"] = 0
for index in event_names.index:

    if event_names.loc[index,"Cinco De Mayo_x"] == 1 or event_names.loc[index,"Cinco De Mayo_y"] == 1:

        event_names.loc[index,"Cinco De Mayo"] = 1        

    if event_names.loc[index,"Easter_x"] == 1 or event_names.loc[index,"Easter_y"] == 1:

        event_names.loc[index,"Easter"] = 1    

    if event_names.loc[index,"Father's day_x"] == 1 or event_names.loc[index,"Father's day_y"] == 1:

        event_names.loc[index,"Father's day"] = 1    

    if event_names.loc[index,"OrthodoxEaster_x"] == 1 or event_names.loc[index,"OrthodoxEaster_y"] == 1:

        event_names.loc[index,"OrthodoxEaster"] = 1    

        

event_names.drop('Cinco De Mayo_x', axis=1, inplace=True)

event_names.drop('Cinco De Mayo_y', axis=1, inplace=True)

event_names.drop('Easter_x', axis=1, inplace=True)

event_names.drop('Easter_y', axis=1, inplace=True)

event_names.drop("Father's day_x", axis=1, inplace=True)

event_names.drop("Father's day_y", axis=1, inplace=True)

event_names.drop('OrthodoxEaster_x', axis=1, inplace=True)

event_names.drop('OrthodoxEaster_y', axis=1, inplace=True)
event_names.columns
event_type_one = calendar["event_type_1"]

event_type_one = pd.get_dummies(event_type_one)

event_type_one.head()
event_type_two = calendar["event_type_2"]

event_type_two = pd.get_dummies(event_type_two)

event_type_two.head()
event_types = pd.merge(event_type_one, event_type_two, right_index=True, left_index=True)

event_types['Cultural'] = 0

event_types['Religious'] = 0

event_types.head()
for index in event_types.index:

    if event_types.loc[index,"Cultural_x"] == 1 or event_types.loc[index,"Cultural_y"] == 1:

        event_types.loc[index,"Cultural"] = 1        

    if event_types.loc[index,"Religious_x"] == 1 or event_types.loc[index,"Religious_y"] == 1:

        event_types.loc[index,"Religious"] = 1    

        

        

event_types.drop('Cultural_x', axis=1, inplace=True)

event_types.drop('Cultural_y', axis=1, inplace=True)

event_types.drop('Religious_x', axis=1, inplace=True)

event_types.drop('Religious_y', axis=1, inplace=True)



event_types.head()
calendar = pd.concat([calendar, event_names, event_types], axis=1)

calendar.drop('event_name_1', axis=1, inplace=True)

calendar.drop('event_type_1', axis=1, inplace=True)

calendar.drop('event_name_2', axis=1, inplace=True)

calendar.drop('event_type_2', axis=1, inplace=True)
calendar.head()
calendar.isnull().sum()
calendar.to_csv('calendar_dummied.csv', index=False)