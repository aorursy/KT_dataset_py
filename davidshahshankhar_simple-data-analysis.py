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
game = pd.read_csv('/kaggle/input/list-of-video-games-considered-the-best/video_games_considered_the_best.csv')
game.head()
del game['Unnamed: 0']
game
game.shape
game['genre'].value_counts()
import matplotlib.pyplot as plt

game['genre'].value_counts()[:20].plot(kind='barh')
game
game['plataform'].value_counts()[:20].plot(kind='barh')
game

game['publisher'].value_counts()
game['publisher'].value_counts()[:20].plot(kind='barh')
game['year'].value_counts()
#Best fighting games from the list
fighting_games = game[game['genre'] == 'Fighting']
fighting_games
#Best sports games from the list
sports_games = game[game['genre'] == 'Sports']
sports_games
#Best action-adv games from the list
aa_games = game[game['genre'] == 'Action-adventure']
aa_games
