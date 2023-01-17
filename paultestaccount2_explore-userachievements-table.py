import pandas as pd

import numpy as np

import os

UserAchievements = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv')
UserAchievements.sort_values('UserId',ascending=False).head()
UserAchievements.sort_values('TierAchievementDate',ascending=False).head()