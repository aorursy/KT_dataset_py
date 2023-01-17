import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import accuracy_score

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data source

df = pd.read_csv("../input/Employee_DS.csv")
# showing 5 rows

df.head()
# I dont need some of these columns so I am going to drop them

df.drop(['Company ID', 'Company Website'], axis=1, inplace = True)



# I am going to drop all rows that have a NaN value. This way i'm only working with data that has an actual employee count value

df.dropna(inplace = True)
#  basic stats for each data source - pretty similar overall

print(df['Employee Count 1'].describe())

print(df['Employee Count 2'].describe())
# This will just show how many of our values in each data source match exactly to the actual employee values.

ds1_acc = accuracy_score(df['Actual Employee Count'], df['Employee Count 1'])

ds2_acc = accuracy_score(df['Actual Employee Count'], df['Employee Count 2'])

print(f"Data Source 1: {ds1_acc*100}{'%'} \nData Source 2: {ds2_acc*100}{'%'}")
# Turning some columns into NP arrays

emp1_count = np.array(df['Employee Count 1'])

act_count = np.array(df['Actual Employee Count'])

emp2_count = np.array(df['Employee Count 2'])



# Combining our actual and employee values into a single array for calculation purposes

combine1 = np.column_stack((act_count,emp1_count))

combine2 = np.column_stack((act_count,emp2_count))

accuracy1 = []

accuracy2 = []



# Calculating accuracy properly depending on whether actual or source 1 or 2 is larger

for actual, count1 in combine1:

    if actual > count1:

        z = ((count1/actual)*100)

        accuracy1.append(z)

    elif actual < count1:

        z = (((actual/count1)*100))

        accuracy1.append(z)

    else:

        accuracy1.append(100)



for actual, count2 in combine2:

    if actual > count2:

        z = ((count2/actual)*100)

        accuracy2.append(z)

    elif actual < count2:

        z = (((actual/count2)*100))

        accuracy2.append(z)

    else:

        accuracy2.append(100)

        

        

# Adding accuracy for each data source to our dataframe

df['Accuracy1'] = accuracy1

df['Accuracy2'] = accuracy2





# printing our mean accuracy for each data source

print(df['Accuracy1'].mean())

print(df['Accuracy2'].mean())



# Graphing accuracy percent

plt.bar(['1'], df['Accuracy1'].mean())

plt.bar(['2'], df['Accuracy2'].mean())

plt.xlabel('Data Source')

plt.ylabel('Average Accuracy')

plt.title('Average Accuracy')
# Calculating Absolute Error and adding it to our DataFrame

df['Absolute Error 1'] = abs(df['Actual Employee Count'] -  df['Employee Count 1'])

df['Absolute Error 2'] = abs(df['Actual Employee Count'] -  df['Employee Count 2'])



# Graphing Total Absolute Error 

plt.bar('1', df['Absolute Error 1'].sum())

plt.bar('2', df['Absolute Error 2'].sum())

plt.xlabel('Data Source')

plt.ylabel('Absolute Error')

plt.title('Total Absolute Error')
# Calculating Mean Absolute Error(MAE)

mae1 = metrics.mean_absolute_error(df['Actual Employee Count'], df['Employee Count 1'])

mae2 = metrics.mean_absolute_error(df['Actual Employee Count'], df['Employee Count 2'])



print(mae1, mae2)



# Graphing MAE

plt.bar('1', mae1)

plt.bar('2', mae2)

plt.xlabel('Data Source')

plt.ylabel('Mean Absolute Error')

plt.title('Mean Absolute Error')
# creating a company size column based on actual employee count

df['Company Size'] = [('Large' if x >= 10000 else ('Medium' if 500 < x < 10000 else 'Small')) for x in df['Actual Employee Count']]



# calling .head() just making sure things look normal

df.head()
# Graphing average accuracy within company size for each source

comp_size_acc1 = df.groupby('Company Size')['Accuracy1'].mean()

comp_size_acc2 = df.groupby('Company Size')['Accuracy2'].mean()

labels = ['Small', 'Medium', 'Large']

x = np.arange(len(labels))  

width = 0.35  # the width of the bars





fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, comp_size_acc1, width, label='Source 1')

rects2 = ax.bar(x + width/2, comp_size_acc2, width, label='Source 2')



ax.set_ylabel('Average Accuracy %')

ax.set_title('Average Accuracy Within Company Size')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()
