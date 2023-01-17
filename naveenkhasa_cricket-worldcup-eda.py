# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



Batsman_Data = pd.read_csv('../input/Batsman_Data.csv')

Ground_Data = pd.read_csv('../input/Ground_Averages.csv')

ODI_Scores_Data = pd.read_csv('../input/ODI_Match_Totals.csv')

ODI_Results_Data = pd.read_csv('../input/ODI_Match_Results.csv')

WC_Players_Data = pd.read_csv('../input/WC_players.csv')

Bowler_Data = pd.read_csv('../input/Bowler_data.csv')



Batsman_Data.describe