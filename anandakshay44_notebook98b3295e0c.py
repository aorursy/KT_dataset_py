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
import pandas as pd 

import matplotlib.pyplot as plt
file = "../input/up_res.csv"

data = pd.read_csv(file,sep=",")

data.head()
color = ['orange','k','c','w','r','m','c','y'] 

plt.title("Party vs Seat Contested")

data['party'].value_counts().plot("bar")

data.groupby('party').size()
#votes for bjp per phase

bjp_vote= data.loc[data['party']== 'BJP+']

phase_per_bjp_vote = bjp_vote.groupby('phase')['votes'].sum()

print(phase_per_bjp_vote)

plt.figure(figsize=(8,8))

plt.title("BJP Votes per phase")

plt.bar(phase_per_bjp_vote.index,phase_per_bjp_vote.values,color='orange')

plt.show()

vote_count = data['votes'].sum()

#print(vote_count)

#votes per phase

vote_phase = data.groupby('phase')['votes'].sum()

#print(vote_phase)

plt.figure(figsize=(8,8))

plt.title("Voting per phase")

plt.pie(vote_phase,labels=vote_phase.index,shadow=True)

plt.show()