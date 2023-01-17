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
def split_week_into_game_plays(week_file):

    """

    Given the week number, open that csv file, and split the file into

      the individual game_play ids

    I used this function to help split up the tracking data into their individual games and plays, which I then used

      in the visualization webapp I've shared

    """

    

    fldr = week_file.split('/')[-1][:-4]

    df_week_file = pd.read_csv(week_file)

    df_week_file['uniqPlayId'] = df_week_file.apply(lambda x: str(x['gameId']) + '_' + str(x['playId']), axis=1)

    uniqPlayIds = list(set(df_week_file['uniqPlayId']))

    for uniqPlayId in uniqPlayIds:

        df_filtered = df_week_file[df_week_file['uniqPlayId']==uniqPlayId]

        df_filtered.to_csv(path_bdb + '/' + fldr + '/' + uniqPlayId + '.csv', index=False)

    

path_bdb = '/kaggle/input/nfl-big-data-bowl-2021/'



"""

#This would be run locally, and you would want to create the weekN/ subfolders before, as this code does not do that

for i in range(1,18):

    df_week_file = split_week_into_game_plays(path_bdb + '/week' + str(i) + '.csv')

    print ('Week', str(i), 'Complete')

"""