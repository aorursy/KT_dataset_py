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
google_apps = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
apps_df = pd.DataFrame()

apps_df['App'] = google_apps['App']

apps_df['Last Updated'] = google_apps['Last Updated']

apps_df = apps_df.set_index('App')
apps_df.loc[apps_df['Last Updated'] == '1.0.19'] 
apps_df = apps_df.drop(['Life Made WI-Fi Touchscreen Photo Frame'])

apps_df.reset_index()
# Convert to dateformat

apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])

apps_df.sort_values(by=['Last Updated'], inplace=True)

apps_df = apps_df.reset_index()
start_date = '2017-01-01'

updated = (apps_df['Last Updated'] >= start_date) 

outdated = (apps_df['Last Updated'] <= start_date)





# Create Dataframes for Separated Apps

updated_apps = pd.DataFrame(data=apps_df.loc[updated])

outdated_apps = pd.DataFrame(data=apps_df.loc[outdated])
# Updated Apps

updated_apps.tail()
# Outdated Apps

outdated_apps.head()
print("Outdated app: ", outdated_apps.iloc[0]['App'])

print('Updated App: ',  updated_apps.iloc[-1]['App'])