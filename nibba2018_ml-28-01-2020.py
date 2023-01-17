import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
emp_data = pd.read_csv("../input/emp_data.csv")

emp_data
plt.hist(emp_data['AGE'])

plt.show()
plt.hist(emp_data['BMI'])

plt.show()
plt.hist(emp_data['PROFIT'])

plt.show()
# Setting the positions and width for the bars

pos = list(range(emp_data.shape[0])) 

width = 0.25



# Plotting the bars

fig, ax = plt.subplots(figsize=(10,10))



# Create a bar with pre_score data,

# in position pos,

plt.bar(pos, emp_data['AGE']*100, width, alpha=0.5, color='#85B486', label=emp_data['ID'][0]) 



# Create a bar with mid_score data,

# in position pos + some width buffer,

plt.bar([p + width for p in pos], emp_data['PROFIT'], width, alpha=0.5, color='#0000FF', label=emp_data['ID'][1]) 



# # Create a bar with post_score data,

# # in position pos + some width buffer,

plt.bar([p + width*2 for p in pos], emp_data['SALES'], width, alpha=0.5, color='#FFA500', label=emp_data['ID'][2]) 



# Set the y axis label

ax.set_ylabel('Stuff')

ax.set_xlabel('Employee ID')



# Set the chart's title

ax.set_title('Employee DATA')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(emp_data['ID'])



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(emp_data['SALES'])+100])



# Adding the legend and showing the plot

plt.legend(['AGE', 'PROFIT', 'SALES'], loc='upper left')

plt.grid()

plt.show()
plt.pie(emp_data['AGE'], explode=[0.5,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], labels=emp_data['ID'])

plt.show()
explode = [0.1 for i in range(0,12)]



explode[emp_data['PROFIT'].idxmax()] = 0.5

explode[emp_data['PROFIT'].idxmin()] = 0.5



plt.pie(emp_data['PROFIT'], explode=explode, labels=emp_data['ID'])

plt.show()
q1 = np.percentile(emp_data['SALES'], 25)

q3 = np.percentile(emp_data['SALES'], 75)



IQR = q3 - q1



least = q1 - 1.5*q3

maxi = q3 + 1.5*q3
emp_data.loc[emp_data['SALES'] < least]
emp_data.loc[emp_data['SALES'] > maxi]
emp_data.drop(emp_data[emp_data['SALES'] < least].index, inplace=True)

emp_data.drop(emp_data[emp_data['SALES'] > maxi].index, inplace=True)