#Q1.pandas program to select the specified columns and rows from a given data frame

import pandas as pd

import numpy as np



exam_data  = {'name': ['Alpha', 'Beeta', 'Gamma', 'Theta', 'Delta', 'Epsilon', 'Rhow', 'Iota', 'Kappa', 'Zeta'],

        'score': [22.9, 37.27, 27.36, 17.89, np.nan, 23.80, 19.5, np.nan, 18.37, 19.79],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6], [1, 3]])
#Q2.Store height of 50 srudents in inches.Now while the data was being recorded manually there has been some typing mistakes therefore height of two students has been recorded as 172 inches and 2 students recorded as 12 inches. Graphically plot and show how you can seggregate the normal data from the abnormal data.

import matplotlib.pyplot as plt

heights=[72,71,56,45,67,89,54,58,67,77,77,78,77,73,73,172,72,71,56,

         45,67,89,54,58,67,172,77,78,77,73,73,172,12,54,64,75,75,77,

         88,66,70,12,54,64,75,75,77,88,66,70]

def plot_his(heights):

    start=min(heights)-min(heights)%10

    end=max(heights)+10

    bins=list(range(start,end,5))

    plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='c')

    plt.xlabel('height of students (inches)')

    plt.ylabel('No.of Students')

    plt.show()

print('Total Data')

plot_his(heights)

heights=list(filter(lambda x: not x==172 and not x==12, heights))

print('Normal Data')

plot_his(heights)
#Q3.get the number of observations, missing values and nan values.

test_data = pd.read_csv("../input/titanicdataset-traincsv/train.csv")

print("No.of Observations are:")

print(test_data.count().sum())

print("No. of Nan is:")

print(test_data.isnull().sum())