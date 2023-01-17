# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/champs-scalar-coupling/champs-scalar-coupling-publicleaderboard.csv')

data.head()
import datetime



start = datetime.datetime.strptime("29-05-2019", "%d-%m-%Y")

end = datetime.datetime.strptime("29-06-2019", "%d-%m-%Y")

date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

date_generated = [date_generated[x].date() for x in range(0, len(date_generated)-1)]
all_mins=[]

for x in date_generated:

    dataS=data.loc[data['SubmissionDate'] <= str(x+ datetime.timedelta(days=1))]

    min=dataS.min()[3]

    all_mins.append(min)
all_mins=np.array(all_mins)
import matplotlib.pyplot as plt

import seaborn as sns
y_pos = np.arange(len(date_generated))

plt.figure(figsize=(20,7))

plt.bar(y_pos,all_mins,color=(0.5, 0.4, 0.6, 0.2))

plt.ylabel('Best score')

plt.xticks(y_pos, date_generated, rotation=90,color=(0.8, 0.4, 0.6, 1))

plt.yticks(color=(0.8, 0.4, 0.6, 1))

plt.axvline(x=15.5)


