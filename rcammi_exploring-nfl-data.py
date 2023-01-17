# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
nfl_df = pd.read_csv("../input/NFLPlaybyPlay2015.csv", index_col=0)
nfl_df.head()
nfl_df.dtypes
# What different values are in the FieldGoalResult column

nfl_df.FieldGoalResult.unique()
nfl_df.PlayType.unique()
fieldgoals = nfl_df[nfl_df.PlayType == "Field Goal"]
fieldgoals.FieldGoalResult.unique()
sns.countplot(x="FieldGoalResult", data=fieldgoals);
sns.barplot(x="FieldGoalResult", y="FieldGoalDistance", data=fieldgoals);
fieldgoals[["posteam", 

            "FieldGoalResult", 

            "FieldGoalDistance", 

            "ScoreDiff",

            "qtr",

            "down",

            "time",

            "TimeUnder",

            "TimeSecs"

           ]].head()
sns.boxplot(x="FieldGoalResult", y="FieldGoalDistance", data=fieldgoals);
(fieldgoals[fieldgoals.FieldGoalResult == "Good"].PlayType.size / fieldgoals.PlayType.size)*100