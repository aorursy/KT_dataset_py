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
def global_data():

    '''Combining three types of data sets Confirmed, deaths, recovered daywise counts

    Columns: 'Province/State', 'type', 'Country/Region', 'Lat', 'Long', remaining dates like '1/22/20', '1/23/20'

    type: 'confirmed', 'deaths', 'recovered' '''



    data = pd.DataFrame()

    base = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_'

    for variable in ['confirmed', 'deaths', 'recovered']:

        path = base + variable + '_global.csv'

        data_part = pd.read_csv(path)

        data_part.insert(4, 'condition', [variable]*len(data_part))

        data_part['type'] = variable

        data = data.append(data_part, ignore_index=True)

    return data
def india_data(no_files):

    ''' At present there are 4 datasets 

        and they may further split/add datasets, 

        check bellow link for dataset updates

        https://api.covid19india.org/

        columns are ['Age Bracket', 'Backup Notes',

       'Contracted from which Patient (Suspected)', 'Current Status',

       'Date Announced', 'Detected City', 'Detected District',

       'Detected State', 'Entry_ID', 'Estimated Onset Date', 'Gender',

       'Nationality', 'Notes', 'Num Cases', 'Num cases', 'Patient Number',

       'Source_1', 'Source_2', 'Source_3', 'State Patient Number',

       'State code', 'Status Change Date', 'Type of transmission'] '''

    data = pd.DataFrame()

    for i in range(1, no_files+1):

        data_part = pd.read_csv('http://api.covid19india.org/csv/latest/raw_data' + str(i) + '.csv')

        data = data.append(data_part, ignore_index=True)

    data = data.sort_values('Detected State')

    return data
india_data = india_data(no_files=4)

global_data = global_data()
