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
# get the game events for jsnell19, round 2:

# all game events

game_events = pd.read_csv("../input/terra-mystica/game_events.csv")



# jsnell19 game events

jsnell19 = game_events[game_events['game'] == 'jsnell19']



# round 2 game events

js19r2 = jsnell19[jsnell19['round'] == 2]



# turn 3

js19r2t3 = js19r2[js19r2['turn'] == 3]



print(js19r2t3)