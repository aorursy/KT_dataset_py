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
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv')
df.head()
df.info()
df[df.Genre.isnull()]
g1 = sns.catplot(x = 'Genre', data = df, kind = 'count', height = 9, aspect = 1.5)
plt.show(g1)
g2 = sns.catplot(x = 'Year_of_Release', data = df, kind = 'count', height= 12, aspect =2)
plt.xticks(rotation = 30)
plt.show(g2)
g3 = sns.catplot(x = 'Platform', data = df, kind = 'count', height= 12, aspect =2)
plt.xticks(rotation = 30)
plt.show(g3)
df_nintendo = df[df.Publisher == 'Nintendo']
order = ['NES','SNES','GB','N64','GBA','GC','DS','3DS','Wii','WiiU']
g4 = sns.catplot(x = 'Platform', data = df_nintendo, kind = 'count', hue = 'Year_of_Release', height = 12, aspect = 2, order = order)
plt.title('Games published by Nintendo on the different Nintendo platforms')
plt.xticks(rotation = 30)
plt.show(g4)
sns.pairplot(data = df_nintendo, kind = 'reg')
plt.show()
df.info()
df2 = df.drop(["Critic_Score","Critic_Count","User_Score","Developer","Rating"], axis = 1)
X = df2.drop(["Global_Sales", "Name"], axis = 1)
y = df2["Global_Sales"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, test_size = 0.25)


