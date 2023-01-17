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
data = pd.read_csv("/kaggle/input/golden-globe-awards/golden_globe_awards.csv")
nominee_data = data.groupby(by='nominee')['year_film']

nominee_data = nominee_data.agg(['mean', 'count']).sort_values('count', ascending=False)[['count']]

# nominee_win_data = nominee_win_data.rename(columns={'wins':'count'})

nominee_data.columns=['nominations']

nominee_data
nominee_win_data = data[data['win'] == True].groupby(by='nominee')['year_film']

nominee_win_data = nominee_win_data.agg(['mean', 'count']).sort_values('count', ascending=False)[['count']]

# nominee_win_data = nominee_win_data.rename(columns={'wins':'count'})

nominee_win_data.columns=['wins']

nominee_win_data
all_nominee_data = pd.concat([nominee_data, nominee_win_data], axis=1)
all_nominee_data['win_percentage'] = (all_nominee_data['wins']/all_nominee_data['nominations'])*100
all_nominee_data.sort_values('win_percentage', ascending=True).head(10)
all_nominee_data[all_nominee_data['nominations'] > 15].sort_values('win_percentage', ascending=False).head(10)