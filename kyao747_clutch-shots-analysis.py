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



#load raw data

raw = pd.read_csv("../input/shot_logs.csv")    
#look at the data of the shots attempted under the following tight/clutch conditions

counts = raw[(raw.FINAL_MARGIN < 5) & (raw.GAME_CLOCK < '0:04') & (raw.PERIOD == 4)].groupby('SHOT_RESULT').size()

print(counts)

#This is such a small dataset, but we do what we can.  

print("Shotting % under clutch conditions = " + str(counts['made'] / counts.sum()))

print('\n')

#let's look at the overall rates

counts_all = raw.groupby('SHOT_RESULT').size()

print(counts_all)

print("Shotting % under regular conditions = " + str(counts_all['made'] / counts_all.sum()))

#to be continued