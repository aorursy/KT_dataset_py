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
import pandas as pd

import numpy as np

laliga = pd.read_csv("../input/Laliga.csv")
laliga
laliga.info()
laliga.replace('-',np.nan, inplace = True)

laliga = laliga.fillna(0)

laliga
laliga['Debut'] = laliga['Debut'].astype(str).str[:4].astype(int)

laliga_new = laliga[laliga['Debut'].between(1930, 1980)]

laliga_new
laliga['Points'] = laliga.Points.astype(float)

laliga.sort_values(by=['Points'],ascending = False)
laliga['GoalsFor'] = laliga.GoalsFor.astype(float)

laliga['GoalsAgainst'] = laliga.GoalsAgainst.astype(float)

laliga['GoalsDifferences']=laliga['GoalsFor']-laliga['GoalsAgainst']
laliga.head(1)
laliga.tail(1)
laliga['GamesWon'] = laliga.GamesWon.astype(float)

laliga['GamesPlayed'] = laliga.GamesPlayed.astype(float)

laliga['WinningPercent'] = (laliga.iloc[:,5]/laliga.iloc[:,4])*100

laliga
laliga.groupby(laliga.iloc[:,19]).sum()