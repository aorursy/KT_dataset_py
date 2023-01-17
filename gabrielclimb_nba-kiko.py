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
nba = pd.read_csv("/kaggle/input/nba-players-data/all_seasons.csv", index_col=0)

nba.head()
nba.season.unique()
nba.groupby(['season', 'pts']).max().player_name
nba.merge(right=nba.groupby(['season']).agg(max_pts=('pts', 'max')).reset_index(), how='inner',

          left_on=['season', 'pts'], right_on=['season', 'max_pts'])
nba_96_97 = nba.query("season=='1996-97'")
# usando idxmax

nba_96_97.iloc[nba_96_97.pts.idxmax(), :]
# usando max

nba_96_97[nba_96_97.pts == nba_96_97.pts.max()]