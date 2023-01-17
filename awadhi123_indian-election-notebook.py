# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting and visulaizing data
national_data = pd.read_csv("/kaggle/input/indian-election-dataset/indian-national-level-election.csv")
national_data.head(5)    #view top 5 rows of dataframe
national_data.shape    #gives (number_of_rows , number_of_column)
national_data.describe()
national_data.dtypes    
national_data.isna().sum()    # pc_type has 8070 and cand_sex has 542 missing or null values
national_data.columns   #gives all column name
data_2014 = national_data[national_data['year'] == 2014]
up_data_2014 = data_2014[data_2014['st_name'] =='Uttar Pradesh']
up_data_2014_max_voted_party = up_data_2014.groupby('partyabbre')['totvotpoll'].sum()
up_data_2014_max_voted_party = pd.DataFrame(up_data_2014_max_voted_party)
up_data_2014_max_voted_party = up_data_2014_max_voted_party.sort_values(by= 'totvotpoll',ascending=False)
up_data_2014_top5_max_voted_party = up_data_2014_max_voted_party.head(5)
up_data_2014_top5_max_voted_party = up_data_2014_top5_max_voted_party.reset_index()
up_data_2014_top5_max_voted_party 
plt.figure(figsize = (12,5))
plt.bar(up_data_2014_top5_max_voted_party['partyabbre'],up_data_2014_top5_max_voted_party['totvotpoll'], color=[ 'red', 'green', 'blue', 'cyan', 'yellow'])
data_2014 = national_data[national_data['year'] == 2014]
mh_data_2014 = data_2014[data_2014['st_name'] =='Maharashtra']
mh_data_2014_max_voted_party = mh_data_2014.groupby('partyabbre')['totvotpoll'].sum()
mh_data_2014_max_voted_party = pd.DataFrame(mh_data_2014_max_voted_party)
mh_data_2014_max_voted_party = mh_data_2014_max_voted_party.sort_values(by= 'totvotpoll',ascending=False)
mh_data_2014_top5_max_voted_party = mh_data_2014_max_voted_party.head(5)
mh_data_2014_top5_max_voted_party 
plt.figure(figsize = (12,5))
plt.bar(up_data_2014_top5_max_voted_party['partyabbre'],up_data_2014_top5_max_voted_party['totvotpoll'], color=[ 'red', 'green', 'blue', 'cyan', 'yellow'])
yearwise_partywise_total_votes = national_data.groupby(['year','partyabbre'])['totvotpoll'].sum()
yearwise_partywise_total_votes = pd.DataFrame(yearwise_partywise_total_votes).reset_index()
years = yearwise_partywise_total_votes['year'].unique()
yearwise_partywise_total_votes_again = {}
for year in list(years):
    yearwise_partywise_total_votes_again['year_' + f'{year}'] = yearwise_partywise_total_votes[yearwise_partywise_total_votes['year'] == year].sort_values(by = 'totvotpoll',ascending= False).head(8)
plt.rcParams.update({'font.size': 22})
plt.figure(figsize = (25,75))
for index,(key,value)  in enumerate(yearwise_partywise_total_votes_again.items()):
    dataframe = yearwise_partywise_total_votes_again[key]
    plt.subplot(11, 1, index+1)
    plt.bar(dataframe['partyabbre'],dataframe['totvotpoll'],color=[ 'red', 'green', 'blue', 'cyan', 'yellow'])
    plt.title(key)
    
