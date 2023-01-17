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
import pandas as pd
df = pd.read_csv('../input/english-premier-leagueepl-player-statistics/pl_19-20.csv')
df
df[df['Position'] == 'Defender']
df[df['Position'] == 'Forward']
df[df['Position'] == 'Midfielder']
df[df['Position'] == 'Goalkeeper']
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd
plt.scatter(df['Goals'], df['Appearances'])
import seaborn as sns
sns.set()
plt.figure(figsize=(15,4))
with sns.axes_style(style='ticks'):
    g = sns.factorplot("Goals", "Appearances", "Position", data=df, kind="box", size=10, aspect=4)
    g.set_axis_labels("Goals", "Appearances");