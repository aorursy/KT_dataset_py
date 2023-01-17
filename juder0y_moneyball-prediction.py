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
base = pd.read_csv("/kaggle/input/moneyball-mlb-stats-19622012/baseball.csv")

base.head()
base.count()
dpyears = base[base.Year < 2002]  

dpyears.head()
dpyears.count() #number of observations decreased because of year range
# check = base[base.Year == 2002]

# teamcount = check['Team'].count()

# teamcount = base[base.Year == 2001][base.Playoffs == 1]['Team'].count()

teamcount = base[base.Year == 2001]['Team'].count()

teamcount
teams = base.Team.value_counts()

teams
qualifiedwins = base[base.Year < 2002]

# qualifiedwins

qualifiedwinsnew = qualifiedwins[['Team','Year','W','Playoffs']]

# qualifiedwinsnew = qualifiedwinsnew[qualifiedwinsnew.Playoffs == 1]

qualifiedwinsnew
import seaborn as sns; 

import matplotlib.pyplot as plt

# tips = sns.load_dataset("qualifiedwinsnew")

plt.figure(figsize=(10,9))

ax = sns.scatterplot(x="W", y="Team", hue="Playoffs",data=qualifiedwinsnew)

plt.plot(95, 0, color='r')
runs = base[base.Year < 2002]

runs.info()
runs['RD'] = runs['RS'] - runs['RA']

runs.info()
sns.lmplot(x ="RD", y ="W", data = runs, order = 2, ci = None)
from sklearn import preprocessing, svm 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 



X = np.array(runs['RD']).reshape(-1, 1) 

y = np.array(runs['W']).reshape(-1, 1) 

  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

  

# Splitting the data into training and testing data 

regr = LinearRegression() 

  

m = regr.fit(X_train, y_train)

m.get_params()
print(regr.score(X_test, y_test))
regr.predict(np.array(133).reshape(-1, 1))
import statsmodels.formula.api as smf

smf.ols(formula ='W ~ RD',data=runs).fit().summary()
# runscored = runs[runs.Year == 2001]

smf.ols(formula ='RS ~ OBP + SLG',data=runs).fit().summary()
obp = runs[runs.Year == 2001][runs.Team == 'OAK'][['Team','OBP','SLG']]

obp
smf.ols(formula ='RA ~ OOBP + OSLG',data=runs).fit().summary()
oobp = runs[runs.Year == 2001][runs.Team == 'OAK'][['Team','OOBP','OSLG']]

oobp