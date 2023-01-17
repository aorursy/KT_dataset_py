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
import numpy as np 
import pandas as pd 
data = pd.read_csv("../input/data.csv")
data.head()
nonull =  data[pd.notnull(data['shot_made_flag'])]
alpha = 0.02
plt.figure(figsize=(7,6))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(nonull.loc_x, nonull.loc_y, color='blue', alpha=alpha)
plt.title('Plot of loc_x against loc_y')
alpha = 0.05
plt.figure(figsize=(5,5))
points = nonull.loc[nonull.shot_made_flag == 1]
plt.scatter(points.loc_x, points.loc_y,color = "#008000", alpha=alpha)
plt.title("Went in, Goodjob Kobe.")

alpha = 0.05
plt.figure(figsize=(5,5))
points = nonull.loc[nonull.shot_made_flag == 0]
plt.scatter(points.loc_x, points.loc_y,color = "#FF0000", alpha=alpha)
plt.title("Missed, Nice try kobe.")

alpha = 0.05
plt.subplot(121)
points = nonull.loc[nonull.shot_made_flag == 1]
plt.scatter(points.loc_x, points.loc_y,color = "#008000", alpha=alpha)
plt.title("Went in")

alpha = 0.05
plt.subplot(122)
points = nonull.loc[nonull.shot_made_flag == 0]
plt.scatter(points.loc_x, points.loc_y,color = "#FF0000", alpha=alpha)
plt.title("Missed")

data_shots = data[data['shot_made_flag']>=0]
data_missed = data_shots.shot_made_flag == 0
data_success = data_shots.shot_made_flag == 1

shot_missed = data_shots[data_missed].season.value_counts()
shot_success = data_shots[data_success].season.value_counts()
shots = pd.concat([shot_success,shot_missed],axis=1)
shots.columns=['Success','Missed']


fig = plt.figure(figsize=(16,5))
shots.plot(ax=fig.add_subplot(111), kind='bar',stacked=False,rot=1,color=['#008000','#FF0000'])
plt.xlabel('Season')
plt.ylabel('Number of shots')
plt.legend(fontsize=15)
