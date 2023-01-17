# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
 #       print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df = pd.read_csv('../input/video-games-sales-2019/vgsales-12-4-2019-short.csv', index_col=['Name'])
game_file_path='../input/video-games-sales-2019/vgsales-12-4-2019-short.csv'
df.head(3)
game_data=pd.read_csv(game_file_path) 
game_data.columns
game_features=['Rank', 'Name', 'Genre','User_Score','Total_Shipped']
df.iloc[4]
df.loc['Wii Sports']
df.loc[['Wii Sports','Halo 3','Portal 2']]
df.iloc[[2,1,0]]
df[:3]
df[3:4]
df[3:15:4]
df['Genre'].head(5)
df.columns=[col.replace(' ','_').lower() for col in df.columns]
print(df.columns)
df[['genre','platform']][:3]
df.genre.iloc[[3]]
df[df.genre == 'Shooter'].head(5)
df[(df.critic_score>7) & (df.year<2019)].head(5)
df[df['genre'].str.split().apply(lambda x: len(x) == 9)].head(3)
df[df.platform.isin(['Wii', 'NES', 'PC'])].head()