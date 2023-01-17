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



# Any results you write to the current directory are saved as output.
deliveriesDF = pd.DataFrame.from_csv('../input/deliveries.csv')
matchesDF = pd.DataFrame.from_csv('../input/matches.csv')

matchesDF['wonwithtoss'] = np.where((matchesDF['toss_winner'] == matchesDF['winner']), True, False)



counts = pd.DataFrame({'count' : matchesDF.groupby( [ "winner", "wonwithtoss"] ).size()}).reset_index()

counts.groupby(['wonwithtoss']).sum().unstack().plot(title='Winners who won the toss', kind='pie')