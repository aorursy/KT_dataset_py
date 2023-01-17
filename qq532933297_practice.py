# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

allData = pd.read_csv('../input/data.csv')
allData.drop('team_id',axis=1,inplace=True)
allData.drop('team_name',axis=1,inplace=True)
data = allData[allData['shot_made_flag'].notnull()].reset_index()
