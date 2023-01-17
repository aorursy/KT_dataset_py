# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ufcdata/data.csv')

#verify date and time type
df["date"] =  pd.to_datetime(df['date'])

#filter between dates
start_date = '06-09-2014'
end_date = '06-09-2019'
mask = (df['date'] > start_date) & (df['date'] <= end_date)

#extract rows with dates between start and end date
df = df.loc[mask]

#store the number of appearences of a fighters name
#the total number of wins equal to the total count of the fighter
red_winners = []
blue_winners = []

for index in range(0,len(df)):
    if df['Winner'][index] == "Red":
        red_winners.append(df['R_fighter'][index])
    else:
        blue_winners.append(df['B_fighter'][index])

red_df = pd.Series(red_winners)
blue_df = pd.Series(blue_winners)

#combine both series and get the unique counts
final_df = pd.concat([red_df,blue_df],axis=0)

#display top 5 winners
final_df.value_counts().head()
        


