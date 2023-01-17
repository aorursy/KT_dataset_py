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
import gc
import dask.dataframe as ddf
import os
FILEPATH = '/kaggle/input/fifa19/data.csv'
def get_filesize(filepath):
    
    file_bytes = os.stat(filepath).st_size / 1024
    
    file_kb = file_bytes / (1024)
    
    file_mb = file_bytes / (1024 * 1024)
    
    return file_kb
get_filesize(FILEPATH)
def get_data_pandas(filepath = FILEPATH):
    
    df2 = pd.read_csv(filepath)
    
    return df2
%time
df_pandas = get_data_pandas(FILEPATH)
gc.collect()
df_pandas.head()
# df_pandas.columns
dtypes = {
    'Unnamed: 0' : 'object', 
    'ID' : 'object', 
    'Name' : 'object', 
    'Age' : 'object', 
    'Photo' : 'object', 
    'Nationality' : 'object', 
    'Flag' : 'object',
    'Overall' : 'object', 
    'Potential' : 'object', 
    'Club' : 'object', 
    'Club Logo' : 'object', 
    'Value' : 'object', 
    'Wage' : 'object', 
    'Special' : 'object',
    'Preferred Foot' : 'object', 
    'International Reputation' : 'object', 
    'Weak Foot' : 'object',
    'Skill Moves' : 'object', 
    'Work Rate' : 'object', 
    'Body Type' : 'object', 
    'Real Face' : 'object', 
    'Position' : 'object',
    'Jersey Number' : 'object', 
    'Joined' : 'object', 
    'Loaned From' : 'object', 
    'Contract Valid Until' : 'object',
    'Height' : 'object', 
    'Weight' : 'object', 
    'LS' : 'object', 
    'ST' : 'object', 
    'RS' : 'object', 
    'LW' : 'object', 
    'LF' : 'object', 
    'CF' : 'object', 
    'RF' : 'object', 
    'RW' : 'object',
    'LAM' : 'object', 
    'CAM' : 'object', 
    'RAM' : 'object', 
    'LM' : 'object', 
    'LCM' : 'object', 
    'CM' : 'object', 
    'RCM' : 'object', 
    'RM' : 'object', 
    'LWB' : 'object', 
    'LDM' : 'object',
    'CDM' : 'object', 
    'RDM' : 'object', 
    'RWB' : 'object', 
    'LB' : 'object', 
    'LCB' : 'object', 
    'CB' : 'object', 
    'RCB' : 'object', 
    'RB' : 'object', 
    'Crossing' : 'object',
    'Finishing' : 'object', 
    'HeadingAccuracy' : 'object', 
    'ShortPassing' : 'object', 
    'Volleys' : 'object', 
    'Dribbling' : 'object',
    'Curve' : 'object', 
    'FKAccuracy' : 'object', 
    'LongPassing' : 'object', 
    'BallControl' : 'object', 
    'Acceleration' : 'object',
    'SprintSpeed' : 'object', 
    'Agility' : 'object', 
    'Reactions' : 'object', 
    'Balance' : 'object', 
    'ShotPower' : 'object',
    'Jumping' : 'object', 
    'Stamina' : 'object', 
    'Strength' : 'object', 
    'LongShots' : 'object', 
    'Aggression' : 'object',
    'Interceptions' : 'object', 
    'Positioning' : 'object', 
    'Vision' : 'object', 
    'Penalties' : 'object', 
    'Composure' : 'object',
    'Marking' : 'object', 
    'StandingTackle' : 'object', 
    'SlidingTackle' : 'object', 
    'GKDiving' : 'object', 
    'GKHandling' : 'object',
    'GKKicking' : 'object', 
    'GKPositioning' : 'object', 
    'GKReflexes' : 'object', 
    'Release Clause' : 'object'    
}
def get_data_dask(filepath = FILEPATH, dtypes = None):
    
    df1 = ddf.read_csv(filepath, dtype = dtypes)
    
    df1 = df1.compute()
    
    return df1
%time
df_dask = get_data_dask(FILEPATH, dtypes)
gc.collect()
df_dask.shape
results = pd.DataFrame(columns = ['Rows', 'Cols', 'Time Taken'])
results.loc['Fifa Dataset - Dask']   = (18207, 89, '7.39 µs')
results.loc['Fifa Dataset - Pandas'] = (18207, 89, '9.06 µs')
results
