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
playlist = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayList.csv")

injury=pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv")

playertrack = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv")
playlist.info()

injury.info()


import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(injury.BodyPart)

plt.legend()

plt.title("Injured Body Parts")

#lmplot()


