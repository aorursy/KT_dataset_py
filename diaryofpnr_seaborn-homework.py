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
#This is my first experience with Seaborn.So it contains very basic examples.
#I will improve as i get more experience
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

video_game_sales=pd.read_csv("../input/vgsales.csv",encoding="windows-1252")
video_game_sales.head()

video_game_sales.info()
sns.jointplot(x='Global_Sales',y='Other_Sales',data=video_game_sales,kind='reg',size=5, ratio=3, color="r")
plt.show()
sns.barplot(x="Global_Sales", y="Genre", data=video_game_sales)
plt.figure(figsize=(15,10))
plt.show()


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(video_game_sales.corr(), annot=True, linewidths=0.9,linecolor="red", fmt= '.1f',ax=ax)
plt.show()