# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
shot_made_flag_list = list()
shot_made_flag_list = (0,1)
data = data[data['shot_made_flag'].isin(shot_made_flag_list)]
data = data.sort_values(by=['game_date','period','minutes_remaining','seconds_remaining'], ascending= [True, True, False,False])

group_action_type = data.groupby("action_type")
action_type_keys = group_action_type.groups.keys()
for x in action_type_keys:
    data[x] = 0

#for y in action_type_keys:
#    data[y] = 1


#cols = ['action_type', 'Jump Shot', 'Layup Shot']
#data[cols]

data.loc[22901]


data["action_type"]
