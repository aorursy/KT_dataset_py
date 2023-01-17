# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

pd.set_option('display.max_rows', 20)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
df
df.Critic_Score.plot.hist(bins=11, grid=False, rwidth=0.95); 
df.plot.scatter("Critic_Score", "Global_Sales", alpha=0.3);
df.groupby("Genre")["Critic_Score"].mean()
df.sort_values("Global_Sales", ascending=False)