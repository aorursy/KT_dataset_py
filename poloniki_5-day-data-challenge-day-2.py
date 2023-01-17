import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# The task was to use .describe 

data = pd.read_csv('../input/degrees-that-pay-back.csv')

#Actual Task

data.describe()
data.head()
#Exploring Columns

print(data.info())
import re

#pattern = re.compile('\$\d{2}\,\d{3}\.\d{2}')

#result = pattern.match('$10,000.00')

#print(bool(result))
pattern = re.compile(r'\$\d{1,5}[\.\,]\d{2,5}\.\d{2}|\d{2,3}\.\d{1,2}')

result = pattern.match('$400,000.00')

print(bool(result))
def dollar(row, pattern):

    sms = row['Mid-Career Median Salary']

    

    if bool(pattern.match(sms)):

            sms = sms.replace('$', '')

            sms = sms.replace('.00', '')

            sms = sms.replace(',', '')

            sms = int(sms)

            

    return sms

data['Mid Median'] = data.apply(dollar, axis=1, pattern=pattern)

data.info()
%matplotlib inline 

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15 , 8))

hist = data['Percent change from Starting to Mid-Career Salary']



# Histogram

plt.hist(hist, bins=7, color='grey', edgecolor='white')

#Customizations

plt.xlabel('Salaries % change distribution')

plt.ylabel('Most frequent %')

plt.title('Salaries increase [in %] from Starting to Mid-Carreer')

plt.xticks([20,30,40,50,60,70,80,90,100], ['20%','30%','40%','50%', '60%', '70%','80%','90%','100%'])

plt.axvspan(51, 87, color='green', alpha=0.4)

plt.axvspan(69, 69.274000, color='red', alpha=1)





# Show and clear plot

plt.show()

plt.clf()