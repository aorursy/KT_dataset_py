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
df_1 = pd.read_csv('../input/blender/1.csv')
df_2 = pd.read_csv('../input/blender/2.csv')
df_3 = pd.read_csv('../input/blender/3.csv')
df_4 = pd.read_csv('../input/blender/4.csv')
df_5 = pd.read_csv('../input/blender/5.csv')

df_1.head()
blended = pd.DataFrame(columns=['id', 'positive', 'negative', 'neutral'])

blended['id'] = df_1['id']

blended['positive'] = np.mean([df_1['positive'], df_2['positive'], df_3['positive'], 
                               df_4['positive'], df_5['positive']], axis=0)

blended['negative'] = np.mean([df_1['negative'], df_2['negative'], df_3['negative'], 
                               df_4['negative'], df_5['negative']], axis=0)

blended['neutral'] = np.mean([df_1['neutral'], df_2['neutral'], df_3['neutral'], 
                               df_4['neutral'], df_5['neutral']], axis=0)
blended.to_csv('blendor.csv', index=False)
