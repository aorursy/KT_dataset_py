# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#parsing the datafiles for each year

file_2015 = '../input/Q1_2015.csv' #year 2015

file_2016 = '../input/Q1_2016.csv' #year 2016

file_2017 = '../input/Q1_2017.csv' #year 2017



 #read the files using pandas

data_15 = pd.read_csv(file_2015)

data_16 = pd.read_csv(file_2016)

data_17 = pd.read_csv(file_2017)



#sum each column in the dataframe

sum_15 = np.sum(data_15, axis =0)

sum_16 = np.sum(data_16, axis = 0)

sum_17 = np.sum(data_17, axis = 0)



#get the column names using .head()

print(data_15.head(1), data_16.head(1), data_17.head(1))



#sum the male columns and divide by 1,000,000, do same for the female columns. Division is by 1 million because this 



#is what will be represented on the bar chart

m_sum_15 = (sum_15['F_Male']+ sum_15['S_Male']+sum_15['P_Male'])/1000000

f_sum_15 = (sum_15['F_Female']+ sum_15['S_Female']+sum_15['P_Female'])/1000000

m_sum_16 = (sum_16['F_Male']+ sum_16['S_Male']+sum_16['P_Male'])/1000000

f_sum_16 = (sum_16['F_Female']+ sum_16['S_Female']+sum_16['P_Female'])/1000000

m_sum_17 = (sum_17['F_Male']+ sum_17['S_Male']+sum_17['P_Male'])/1000000

f_sum_17 = (sum_17['F_Female']+ sum_17['S_Female']+sum_17['P_Female'])/1000000



#put the male and female data in lists: males, females

males = (m_sum_15, m_sum_16, m_sum_17)

females = (f_sum_15, f_sum_16, f_sum_17)



#creating the first plot of RSA Membership for the first quarters of 2015-2017

n_groups = 3 #(since there are three groups of data i.e. 2015, 2016 and 2017)

fig, ax = plt.subplots()

index=np.arange(n_groups)

bar_width = 0.35

opacity = 0.8

rects1 = plt.bar(index, males, bar_width, alpha = opacity, color='b', label='Males')

rects2 = plt.bar(index+bar_width, females, bar_width, alpha = opacity, color='r', label='Female Membership')

plt.xlabel('Years')

plt.ylabel('RSA Membership (in \'000000\')')

plt.title('RSA Membership for Q1 2015, 2016 & 2017')

plt.xticks(index+bar_width, ('2015','2016','2017'))

plt.legend(loc='upper right')

plt.tight_layout()

#plt.savefig('RSA_Q1_mem.jpg')

plt.show()



#data for the second graph

gender = ['Male', 'Female']

x = np.arange(len(gender))

plt.xticks(x,gender)

#2017 rsa data: m_sum_17, f_sum_17 (from above)

rsa_q1_2017 = (m_sum_17, f_sum_17) #put them in a list

tot_m = 36363042/1000000 #total male working population (in million) obtained from Nigerian Bureau of Statistics (NBS) website

tot_f = 33107859/1000000 #total female working population

working_pop= (tot_m, tot_f) #put in a list

y_min = min(rsa_q1_2017) + 1 # this will be the minimum number on the y_axis

y_max = (max(working_pop)+10)

plt.ylim((y_min, y_max))

plt.xlabel('Gender')

plt.ylabel('Working population VS RSA Membership')

plt.title('Q1 2017 RSA membership vs Total Working Population')

plt.plot(x, working_pop, 'red', label='Working Population', marker='o') #line chart

plt.bar(x, rsa_q1_2017, label='2017 RSA Q1 Membership') #bar chart

#plt.savefig('2017 plots.jpg')

plt.show()



#for the third stage i.e. plotting to know the age group for Q1 2017 with the highest RSA account

data_17_r = pd.read_csv(file_2017, index_col=0) #parse the data again, this time with the index column

sum_age =np.sum(data_17_r, axis=1) #sum row wise i.e by age brackets

age_groups = ['Below 30', '30-39', '40-49', '50-59', '60-65', 'Above 65']

x = np.arange(len(age_groups))

y = list(sum_age)

sum_age = np.sum(data_17_r, axis=1)

sum_age = sum_age/100000 #this time dividing by 100000

len(sum_age) == len(age_groups) #just to check that the data in the sum is equal to the defined age groups

x = np.arange(len(sum_age))

plt.xticks(x, age_groups)

y_min = min(sum_age) - 1

y_max = max(sum_age) + 1

plt.ylim((y_min, y_max))

plt.xlabel('Age Groups')

plt.ylabel('RSA Membership')

plt.title('Q1 2017 RSA Membership by Age group')

plt.bar(x, sum_age, label='RSA Membership')

#plt.savefig('Mem_age.jpg')

plt.show()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.