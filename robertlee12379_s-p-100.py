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
#Lists



#Stocks in the S&P 100 are selected to represent sector balance and market capitalization. 

#To begin, let's take a look at what data we have associated with S&P companies.



#Read the dataset

dataset = pd.read_csv('../input/sp 100.txt')



#Explore the dataset

print('Explore the dataset')

print(dataset.head())

print('\n')



#Four lists, names, prices, earnings, and sectors, are available. 

names = list(dataset['Name'])

prices = list(dataset['Price'])

earnings = list(dataset['EPS'])

sectors = list(dataset['Sector'])



#First four items of names

print('First four items of names')

print(names[0:4])

print('\n')



#Print information on last company

print('Print information on last company')

print(names[-1])

print(prices[-1])

print(earnings[-1])

print(sectors[-1])
#NumPy is a scientific computing package in Python that helps you to work with arrays. 

#Let's use array operations to calculate price to earning ratios of the S&P 100 stocks.



# Convert lists to arrays

prices_array = np.array(prices)

earnings_array = np.array(earnings)



# Calculate P/E ratio 

pe = prices_array / earnings_array

print('Calculate P/E ratio')

print(pe)
#Filtering arrays

#Let's focus on two sectors:



#Information Technology

#Consumer Staples



#numpy is imported as np and S&P 100 data is stored as arrays: names, sectors, and pe (price to earnings ratio).

names = np.array(names)

sectors = np.array(sectors)

pe = np.array(pe)



#Create boolean array 

boolean_array = (sectors == 'Information Technology')



# Subset sector-specific data

it_names = names[boolean_array]

it_pe = pe[boolean_array]



# Display sector names

print('Information Technology')

print('Display sector names')

print(it_names)

print('Display P/E ratios')

print(it_pe)

print('\n')



#Create boolean array 

boolean_array = (sectors == 'Consumer Staples')



# Subset sector-specific data

cs_names = names[boolean_array]

cs_pe = pe[boolean_array]



# Display sector names

print('Consumer Staples')

print('Display sector names')

print(cs_names)

print('Display P/E ratios')

print(cs_pe)
#Summarizing sector data

#calculate the mean and standard deviation of P/E ratios for Information Technology and Consumer Staples sectors



it_pe_mean = np.mean(it_pe)

it_pe_std = np.std(it_pe)



print('The mean and standard deviation of P/E ratios for Information Technology')

print(it_pe_mean)

print(it_pe_std)

print('\n')



cs_pe_mean = np.mean(cs_pe)

cs_pe_std = np.std(cs_pe)



print('The mean and standard deviation of P/E ratios for Consumer Staples')

print(cs_pe_mean)

print(cs_pe_std)
#Plot P/E ratios

#Let's take a closer look at the P/E ratios using a scatter plot for each company in these two sectors.

import matplotlib.pyplot as plt



# Make a scatterplot

plt.scatter(range(0,15), it_pe, color='red', label='IT')

plt.scatter(range(0,12), cs_pe, color='green', label='CS')



# Add legend

plt.legend()



# Add labels

plt.xlabel('Company ID')

plt.ylabel('P/E Ratio')

plt.show()
#Histogram of P/E ratios

#To visualize and understand the distribution of the P/E ratios in the IT sector, you can use a histogram.



# Plot histogram 

plt.hist(it_pe, bins=8)



# Add x-label

plt.xlabel('P/E ratio')



# Add y-label

plt.ylabel('Frequency')



# Show plot

plt.show()
#Name the outlier

#We've identified that a company in the Industrial Technology sector has a P/E ratio of greater than 50. Let's identify this company.



# Identify P/E ratio within it_pe that is > 50

outlier_price = it_pe[it_pe > 50]



# Identify the company with PE ratio > 50

outlier_name = it_names[it_pe > 50]



# Display results

print("In 2017, " + str(outlier_name[0]) + " had an abnormally high P/E ratio of " + str(round(outlier_price[0], 2)) + ".")