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
#Load Libraries
import numpy
from numpy import arange
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import read_csv
from pandas import read_excel
filename = ("../input/data-science-social-groups-survey/DataScienceSocialGroupsSurvey.xlsx")
names = ['S.no','S.no','GroupName', 'GroupLink', 'Members', 'SocialMedia', 'ContentType', 'Moderated', 'Level', 'Type', 'InteractionScore', 'ContentScore', 'ModerationScore', 'AverageRating']
data = read_excel(filename,sheet_name="Survey", names=names )
df=pd.DataFrame(data)
df.head(15)
fig, ax= plt.subplots(figsize=(20,10))
x=df['GroupName']
y=df['AverageRating']
ax.barh(x,y, color='orange')
ax.set_title('DataScience Groups Ratings')
ax.set_xlabel('Average Rating')
plt.tight_layout()
fig, ax= plt.subplots(figsize=(20,10))
x=df['GroupName']
z=df['InteractionScore']
ax.barh(x,z, color='green')
ax.set_title('DataScience Groups Ratings on Interaction')
ax.set_xlabel('Interaction Score')
plt.tight_layout()
fig, ax= plt.subplots(figsize=(20,10))
x=df['GroupName']
w=df['ContentScore']
ax.barh(x,w, color='Red')
ax.set_title('DataScience Groups Ratings on Content')
ax.set_xlabel('ContentScore')
plt.tight_layout()
fig, ax= plt.subplots(figsize=(20,10))
x=df['GroupName']
u=df['ModerationScore']
ax.barh(x,u, color='Blue')
ax.set_title('DataScience Groups Ratings on Moderation')
ax.set_xlabel('Moderation Score')
plt.tight_layout()