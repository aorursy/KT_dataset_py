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
import pandas as pd

data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data.info()
data['sick'] = data['target']

data['not sick'] = 1 - data['target']
data.groupby('target').agg('count')['age']

#165 people are sick out of 303 people in sample space
data.groupby('sex').agg('sum')[['sick','not sick']].plot(kind = 'bar', stacked = 'True')

#females are likely to be sick(may be due to sampling error)
import matplotlib.pyplot as plt

alpha = data.age[data.target == 1]

beta = data.age[data.target == 0]

plt.hist([alpha,beta],label=['sick', 'not sick'])

plt.legend(loc='upper right')

print ('mean_age_sick = ' + str(alpha.mean()))

print ('mean_age_notsick = ' + str(beta.mean()))
data.groupby('cp').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

# cp value with 0 are less likely to get sick
alpha = data.trestbps[data.target == 1]

beta = data.trestbps[data.target == 0]

plt.hist([alpha,beta],label=['sick', 'not sick'])

plt.legend(loc='upper right')

print ('mean_trestbps_sick = ' + str(alpha.mean()))

print ('mean_trestbps_notsick = ' + str(beta.mean()))

# No clear difference b/w sick & non sick people in terms of resting blood pressure
alpha = data.chol[data.target == 1]

beta = data.chol[data.target == 0]

plt.hist([alpha,beta],label=['sick', 'not sick'], bins = 10)

plt.legend(loc='upper right')

print ('mean_chol_sick = ' + str(alpha.mean()))

print ('mean_chol_notsick = ' + str(beta.mean()))

#not much difference in groups
data.groupby('fbs').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

#people with high fbs are less, no clear conclusion
data.groupby('restecg').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

#people with ecg 1 are more likely to be sick than who have 0, no clear conclusion about 2
alpha = data.thalach[data.target == 1]

beta = data.thalach[data.target == 0]

plt.hist([alpha,beta],label=['sick', 'not sick'], bins = 10)

plt.legend(loc='upper right')

print ('mean_thalach_sick = ' + str(alpha.mean()))

print ('mean_thalach_notsick = ' + str(beta.mean()))

# On an average sick people have higher max heart rate
data.groupby('exang').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

# People with no exercise induced angina are likely to be sick.(counter-intuitive)
alpha = data.oldpeak[data.target == 1]

beta = data.oldpeak[data.target == 0]

plt.hist([alpha,beta],label=['sick', 'not sick'], bins = 10)

plt.legend(loc='upper right')

print ('mean_oldpeak_sick = ' + str(alpha.mean()))

print ('mean_oldpeak_notsick = ' + str(beta.mean()))

# people with 0 oldpeak are highly likely to have heart disease
data.groupby('slope').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

# people with high slope of ST depression are likely to have heart disease 
data.groupby('ca').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')

#people with 0 vessels coloured by flourosopy are likely to have heart disease
data.groupby('thal').agg('sum')[['sick','not sick']].plot(kind = 'bar',stacked = 'True')
