#matplotlib magic code

%matplotlib inline



#import matplotlib 

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



#testing if matplotlib import



plt.plot()

plt.show()







#import the dataset 

recent_grads = pd.read_csv('../input/college-earnings-by-major/recent-grads.csv')

#exploring the recent_grads dataset

print(recent_grads.iloc[0])

print(recent_grads.head())

print(recent_grads.tail())

print(recent_grads.describe())

#find the number of rows in the dataset

raw_data_count = recent_grads.shape[0]



#drop missing na values and asign back to dataframe

recent_grads = recent_grads.dropna()



cleaned_data_count = recent_grads.shape[0]



print('The number of rows before drop na function is ' + str(raw_data_count))

print('The number of rows after the drop na function is ' + str(cleaned_data_count))



#in the intrest in space we do all graphs within this cell.



#scatter plot of Sample_size and Median

recent_grads.plot(x = 'Sample_size', y = 'Median', kind = 'scatter')



#scatter plot of Sample_size and Unemployment_rate

recent_grads.plot(x = 'Sample_size', y = 'Unemployment_rate', kind = 'scatter')



#scatter plot of Full_time and Median

recent_grads.plot(x = 'Full_time', y = 'Median', kind = 'scatter')



#scatter plot of ShareWomen and Unemployment_rate

recent_grads.plot(x = 'ShareWomen', y = 'Unemployment_rate', kind = 'scatter')



#scatter plot of Men and Median

recent_grads.plot(x = 'Men', y = 'Median', kind = 'scatter')



#scatter plot of Women and Median

recent_grads.plot(x = 'Women', y = 'Median', kind = 'scatter')
#Plot multiple histograms using a for loop 



#Save the columns we will be using into a listed variable

colnames = ['Sample_size', 'Median', 'Employed', 'Full_time', 'ShareWomen', 'Unemployment_rate', 'Men', 'Women']



#Save the table names into a listed variable 

table_names = ['Frequency of Sample_size','Frequency of Median','Frequency of Employed','Frequency of Full_time',

               'Frequency of ShareWomen','Frequency of Unemployment_rate','Frequency of Men','Frequency of Women']



#Adjust the fig size to allow more readability 

fig = plt.figure(figsize = (15,20)) # figsize = (width,height)



#A line of code that makes multiple histogram grpahs

for r in range(0,8): 

    ax = fig.add_subplot(4,3,r + 1)

    ax = fig.subplots_adjust(wspace=.4, hspace=.7) #Adjust the space between subplots 

    ax = recent_grads[colnames[r]].plot(kind='hist', rot = 40, edgecolor = 'black', linewidth = 2) #rot = rotation of labels

    ax.set_title(table_names[r], fontsize = 16)

    ax.set_xlabel(colnames[r], fontsize = 13, labelpad = 15) #Labelpad adjust the xlabel space from subplot

    ax.set_ylabel('Frequency', fontsize = 13, labelpad = 15) #labelpad ajust the ylabel from sublot

               
#Create a scatter matrix



#Import the module needed for scatter matrix

from pandas.plotting import scatter_matrix



#Scattter matrix table

#print(scatter_matrix(recent_grads[['Median', 'Total']],figsize = (10,10)))

print(scatter_matrix(recent_grads[['Total', 'Median', 'Major']], figsize = (10,10)))





#Creat a barplot that compares ShareWomen to Majors

recent_grads[:10].plot.bar(x = 'Major', y = 'ShareWomen') # first ten rows

recent_grads[-10:].plot.bar(x = 'Major', y = 'ShareWomen') # last ten rows
#Create a barplot that compares Unemployment_rate to Majors

recent_grads[:10].plot.bar(x = 'Major', y = 'Unemployment_rate') # first 10 rows

recent_grads[-10:].plot.bar(x = 'Major', y = 'Unemployment_rate') # last 10 rows
#find the top ten most popular majors and their median pay

pop_major = recent_grads[['Total','Major','Median']].sort_values(by = 'Total' , ascending = False).head(10)

pop_major
# Find the top ten least popular major and their median pay

unpop_major = recent_grads[['Total','Major','Median']].sort_values(by = 'Total' , ascending = True).head(10)

unpop_major
#Graph top ten most favorite and top ten least favorite

pop_major.plot.bar(x = 'Major', y = 'Median')

unpop_major.plot.bar(x = 'Major', y = 'Median')
print('The average median pay for the top ten most popular majors is ' + str(pop_major['Median'].mean()) + '\n')

print('The average median pay for top ten most unpopular majors is ' + str(unpop_major['Median'].mean()))
#Find all data where major is majority female (Where Female is greater than Male)



#Create a boolean variable to filter the dataset on whether women pop is greater than men pop

female_more_boolean = recent_grads['Women'] > recent_grads['Men']



#Find the number of rows and columns in he original dataset

print('The number of row and columns in the recent_grads dataset are ' + str(recent_grads.shape[0]) + '\n')



#Count how many true values there are within the boolean variable

print('The number of rows where Women have majority population are ' + str(female_more_boolean.value_counts()[1]) + '\n'

      + 'The number of rows where Men have majority population are ' + str(female_more_boolean.value_counts()[0]) + '\n')



#index the boolean variable with recent_grads data set and save it to another variable

female_more_male = recent_grads[female_more_boolean]



#confirm the amount of rows matches the  true boolean value count above

print('The number of (Rows, Columns) are ' + str(female_more_male.shape) + '\n')









#Create a boolean variable to filter the dataset on whether men pop is greater

male_more_boolean = recent_grads['Women'] < recent_grads['Men']



#Count how many true values there are within the boolean variable

print('The number of rows where men have majority population are ' + str(male_more_boolean.value_counts()[1]) + '\n')



#index the boolean variable with recent_grads data set and save it to another variable

male_more_female = recent_grads[male_more_boolean]



#confirm the amount of rows matches the  true boolean value count above

print('The number of (Rows, Columns) are ' + str(male_more_female.shape) + '\n')
#create a histogram that shows the median pay for Women and Men, when Women have majority population in Major

fig = plt.figure(figsize = (10,5))

fig.subplots_adjust(wspace = 1)



ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



ax1.boxplot(female_more_male['Median'])

ax1.set_title('Median Pay for Women Dominated Majors')

ax1.set_ylabel('Median Pay')



ax2.boxplot(male_more_female['Median'])

ax2.set_title('Median Pay for Men Dominated Majors')

ax2.set_ylabel('Median Pay')


