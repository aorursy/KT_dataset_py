# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
raw = pd.read_csv("../input/shot_logs.csv")
shot_n_clock = pd.concat([raw.SHOT_RESULT, raw.SHOT_CLOCK], axis=1)

shot_n_clock["IS_SHOT_MADE"] = (shot_n_clock.SHOT_RESULT == 'made')



#It's interesting to note that there are shots recorded at 0 sec and 24 sec marks

#This suggests some inconsistency in the way data was gathered



#If clock is recorded at the moment of shot-release, then it should always be > 0



#If clock is recorded at the moment of shot made, then it could be 0, but cannot be 24



#Daniel suggested that 24 could be the result of an offensive rebound (mid-air tip-in type of scenario)

#This will require further investigating by incorporating shot distances.



#Never the less, we shall move on ignoring this.

shot_n_clock.SHOT_CLOCK.describe()
#look at the shooting stats for each second shot-clock buckets

clock_tick = list(range(0,25))

trueList = []

falseList = []

for x in clock_tick:

    temp = shot_n_clock[(shot_n_clock.SHOT_CLOCK <= x) & (shot_n_clock.SHOT_CLOCK > x-1)].groupby("IS_SHOT_MADE").SHOT_RESULT.count() 

    trueList.append(temp[1])

    falseList.append(temp[0])      

    

#Building output dataframe, calculating % column    

final = pd.DataFrame({"sec_left": clock_tick, "True": trueList, "False": falseList})

final['FG%'] = np.round(final['True'] / (final['True'] + final['False']) * 100, 2)

print(final)





plt.plot(final['sec_left'], final['FG%'])

plt.show()
#double check the workflow captured all entries

print('number of entries = ', final['True'].sum() + final['False'].sum())
