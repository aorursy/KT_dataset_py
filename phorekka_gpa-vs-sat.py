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
import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
data = pd.read_csv('../input/1.03. Dummies.csv')
data
data.describe()
new_data = data.copy()
new_data['Attendance'] = new_data['Attendance'].map({'Yes':1,'No':0})
new_data.describe()
y = new_data['GPA']

x1 = new_data[['SAT','Attendance']]
x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()

plt.scatter(new_data['SAT'],y,c=new_data['Attendance'],cmap= 'RdYlGn_r')

yhat_yes = 0.8665 + 0.0014*new_data['SAT'] 

yhat_no = 0.6439 + 0.0014*new_data['SAT']

yhat = 0.0017*data['SAT'] + 0.275

fig = plt.plot(new_data['SAT'],yhat_no,c ='green', lw=4, label= 'Regression Line 1')

fig = plt.plot(new_data['SAT'],yhat_yes, c='red', lw =4, label='Regression Line 2')

fig = plt.plot(data['SAT'],yhat, lw=3, c='#4C72B0', label ='regression line')



plt.xlabel('SAT', fontsize = 20)

plt.ylabel('GPA', fontsize = 20)

plt.show()
x.head()
new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendence':[0,1]})
new_data
new_data.rename(index={0:'Alice',1:'Bob'})
prediction = results.predict(new_data)

prediction
predictiondf = pd.DataFrame({'Predictions':prediction})

joined = new_data.join(predictiondf)

joined
joined.rename(index={1:'bob',0:'alice'})