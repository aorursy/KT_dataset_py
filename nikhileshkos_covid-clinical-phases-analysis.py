# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Get the list of the files present.
xml_files = os.listdir('/kaggle/input/covid19-clinical-trials-dataset/COVID-19 CLinical trials studies/COVID-19 CLinical trials studies')
print("Total XML files:- ", len(xml_files))
input_path = '/kaggle/input/covid19-clinical-trials-dataset'
covid_data = pd.read_csv(os.path.join(input_path, 'COVID clinical trials.csv'))
covid_data.head()
# let see the columns values from the datasets

covid_data.columns
# URL Value column is not needed.
covid_data['URL'][:5]

covid_data['Study Results'][:5]
covid_data['Study Documents'].unique
# So, we can remove the url from dataset
covid_updated = covid_data.drop(['URL','Study Results', 'Study Documents','Results First Posted',
                 'Last Update Posted', 'Outcome Measures', 'Study Designs',
                 'First Posted','Other IDs', 'Outcome Measures'], axis = 1)

# we observed the datasets and remove the unnecessary information from the datasets
covid_updated.head(5)
covid_updated.columns
import cufflinks as cf
import plotly.graph_objs as go
cf.go_offline()

covid_updated['Phases'].value_counts().drop('Not Applicable').iplot(kind='bar', color='blue',
                                                                    xTitle='Vaccine Phase Types',
                                                                    yTitle='Number of Candidates')
# we will calculate  all the phases information and drop missing data
# understand more deeply with pie chart

# Get the index
phases_data = covid_updated[covid_updated['Phases'] == 'Phase 4']
phase_index = covid_updated['Phases'].value_counts().drop('Not Applicable').index
phase_values = covid_updated['Phases'].value_counts().drop('Not Applicable').values

fig = go.Figure(data=[go.Pie(labels=phase_index,
                            values=phase_values,
                            textinfo = 'label + percent')])
fig.show()


# Find the final phase candidates

covid_updated['country'] = covid_updated['Locations'].apply(lambda x: str(x).split()[-1])
covid_updated['country'].value_counts().drop('nan').sort_values(ascending=False)[:15].iplot(kind='bar', bins=30,xTitle='Vaccine Clinical Trials', yTitle='Country')
phases_data.head(2)
# phases_data[['Start Date', 'Completion Date']].iplot(kind='bar')

# check which we can get in 2020 and 2021
def get_trial_info(dataframe, time):
    column_name = 'complete_in_{}'.format(time)
    trials = dataframe.loc[phases_data[column_name] == True, 'Completion Date'].index
    for trial_name in trials:
        if str(dataframe.loc[trial_name]['Acronym']).strip() != 'nan' and str(dataframe.loc[trial_name]['country']).strip() != 'nan':
            print("Name: {}: {}: {} ".format(dataframe.loc[trial_name]['Acronym'],
                                                   dataframe.loc[trial_name]['Completion Date'],
                                                  dataframe.loc[trial_name]['country']))

    
    
phases_data['complete_in_2020'] = phases_data['Completion Date'].apply(lambda x: '2020' in str(x))
phases_data['complete_in_2021'] = phases_data['Completion Date'].apply(lambda x: '2021' in str(x))

get_trial_info(phases_data, '2021')

get_trial_info(phases_data, '2020')

