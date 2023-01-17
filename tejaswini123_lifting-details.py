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
basic=pd.read_csv("../input/meets.csv")
basic.head(5)
lifting=pd.read_csv("../input/openpowerlifting.csv")
lifting.head(5)
lifting['Sex'].value_counts()
print(lifting[['Name','Age','BodyweightKg']][lifting.Age == lifting.Age.max()])
print(lifting[['Name','Age','BodyweightKg']][lifting.Age == lifting.Age.min()])
lifting['Equipment'].value_counts()
print(lifting[['Name','Age','BodyweightKg']][lifting.BodyweightKg == lifting.BodyweightKg.max()])
print(lifting[['Name','Age','BodyweightKg']][lifting.BodyweightKg == lifting.BodyweightKg.min()])
df=pd.merge(basic,lifting ,on='MeetID')
df.head(5)
df['MeetCountry'].value_counts()
print(df[['Name','Age','TotalKg']][df.TotalKg==df.TotalKg.max()])
df['Date'].value_counts()
df['MeetName'].value_counts()
df['']