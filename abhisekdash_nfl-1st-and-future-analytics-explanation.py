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
playlist_df = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

PlayerTrackData_df = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')

InjuryRecord_df = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
playlist_df.head()
playlist_df[playlist_df['GameID'] == '26624-1'][0:20]
playlist_df['FieldType'].value_counts()
PlayerTrackData_df.head()
PlayerTrackData_df['event'].value_counts()
InjuryRecord_df.head()
InjuryRecord_df['BodyPart'].value_counts()
InjuryRecord_df[InjuryRecord_df['BodyPart'] == 'Knee']['Surface'].value_counts()
InjuryRecord_df[InjuryRecord_df['BodyPart'] == 'Ankle']['Surface'].value_counts()
InjuryRecord_df['Surface'].unique()
playlist_df.shape
InjuryRecord_df.shape
PlayerTrackData_df.shape
playlist_df[playlist_df['GameID'] == '39873-4']['FieldType'].unique()