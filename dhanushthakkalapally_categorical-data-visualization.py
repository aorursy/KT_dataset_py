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
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np 

sns.set(style='ticks',color_codes=True)
tips = sns.load_dataset('tips')

tips.shape
tips.head()
sns.catplot(x='day',y='total_bill',data = tips,palette=sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(x='day',y='total_bill',data = tips,palette=sns.diverging_palette(220, 20, n=7),jitter=False,size=6);
sns.catplot(kind='swarm',x='day',y='total_bill',data=tips,palette=sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips,palette=['black','#cccdcf'],size=6);
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm",order=['Sat','Thur','Fri','Sun'], data=tips,palette=['black','#cccdcf'],size=6);
sns.catplot(kind='swarm',x='total_bill',y='day',data=tips,palette=sns.diverging_palette(220, 20, n=7),size=6);
# We use kind = 'box' we will get box plot 

sns.catplot(x="day", y="total_bill", kind="box", data=tips,palette=sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(x="day", y="total_bill", kind="box", data=tips,palette=sns.diverging_palette(220, 20, n=7),hue='smoker',size=6);
diamonds = sns.load_dataset('diamonds')

diamonds.shape
diamonds.head()
sns.catplot(kind='boxen',x ='color',y='price',data=diamonds.sort_values('color'),size=6,palette=sns.diverging_palette(220, 20, n=7));
sns.catplot(kind='violin',x='day',y='total_bill',data=tips,palette=sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(kind='violin',x='day',y='total_bill',hue='sex',split=True,inner ='stick',data=tips,palette=sns.diverging_palette(220, 20, n=4),size=6);
g = sns.catplot(kind ='violin',x = 'day',y='total_bill',data=tips,inner=None,palette=sns.diverging_palette(220, 20, n=7),height=6)

sns.swarmplot(x="day", y="total_bill",color='black',size=4, data=tips, ax=g.ax);
sns.catplot(x="day", y="total_bill", kind="bar",data=tips, palette=sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(kind='count',x='day',data=tips,palette = sns.diverging_palette(220, 20, n=7),size=6);
sns.catplot(y="day", hue="sex", kind="count",

            palette=sns.diverging_palette(220, 20, n=3), edgecolor=".6",

            data=tips,size=6);