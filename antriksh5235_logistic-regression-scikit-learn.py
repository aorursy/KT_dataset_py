# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, date, time

from sklearn.linear_model import LogisticRegression

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')



alpha = 10

lamda = 0.001

e = 2.718281828459045235



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
detailed = pd.read_csv('../input/RegularSeasonDetailedResults.csv')

detailed.drop(['Season','Daynum','Wloc','Wteam', 'Lteam', 'Lscore'], axis=1, inplace=True)

detailed.head()

y = pd.read_csv('../input/RegularSeasonDetailedResults.csv', usecols=['Wscore'])

y.head()

#print(list(detailed))
y = np.ravel(y)

clf = LogisticRegression(verbose=1)

clf.fit(detailed, y)



#Prediction Probability

y_pred = clf.predict_proba(detailed)[:,1]



print(zip(y_pred,y))