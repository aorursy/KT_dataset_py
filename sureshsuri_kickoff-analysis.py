import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
vedio_re=pd.read_csv('../input/video_review.csv')
vedio_re.head()
vedio_re.Primary_Impact_Type.value_counts().plot.bar()
vedio_re.groupby(['Player_Activity_Derived', 'Primary_Partner_Activity_Derived'])['Primary_Impact_Type'].value_counts().plot.bar()
vedio_re.groupby(['Player_Activity_Derived', 'Primary_Partner_Activity_Derived'])['GameKey'].value_counts().plot.bar()
vedio_inj=pd.read_csv("../input/video_footage-injury.csv")
vedio_inj.head()

sns.set(style="whitegrid")
g = sns.PairGrid(vedio_inj, y_vars="PlayDescription",
                 x_vars=["season", "gamekey", "playid"],
                 height=5, aspect=.5)

# Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot, scale=1.3, errwidth=4, color="xkcd:plum")
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)

gamedata=pd.read_csv("../input/game_data.csv")
gamedata.head()
