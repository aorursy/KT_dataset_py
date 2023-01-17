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
data = pd.read_csv('../input/HR_comma_sep.csv')

print(data.describe())
#print(data.info)

print(data.head())

print('average_montly_hours vs left')

plt.plot(data['average_montly_hours'],data['left'], 'r')

plt.show()

print('the above shows that people tend to leave when underutillized')
print('satisfaction_level vs left')

plt.plot(data['satisfaction_level'],data['left'], 'r')

plt.show()

print('the above shows that satisfaction level is not a big player to decide if an employee stays back')
print('number_project vs time_spend_company along with a plotting for people left')

plt.plot(data['time_spend_company'],data['number_project'], 'ro',

         data['time_spend_company'],data['left'], 'r+',

         data['left'],data['number_project'], 'r*')

plt.show()

print('We can understand that longer working periods with lots of projects is a small factor for')

print('employees to leave the organization')
sal   = data['salary']

n_sal = [0] * len(sal)

for i in range(len(sal)):

    if   sal[i] == 'low' :

         n_sal[i] = 0

    elif sal[i] == 'medium' :

         n_sal[i] = 1

    else:

         n_sal[i] = 2

            

status = np.array(data['left'])

yexp = np.array(data['time_spend_company'])

nsal = np.array(n_sal)



c_status =['none'] * len(status)

for i in range(len(status)):

    if status[i] == 0 :

        c_status[i] = 'red'

    elif status[i] == 1:

        c_status[i] = 'black'

        

plt.scatter(nsal,yexp,c = c_status)



plt.show()

print('Low Salary may actually not be a contributing factor for him to leave the company')
plt.hist(data['left'])

plt.hist(data['promotion_last_5years'])

plt.show()

print('so people who do not receive a promotion are more prone to leave the company')
print('So from the above study , people who have')

print('1. no career growth , and')

print('2. people who have been underutilised')