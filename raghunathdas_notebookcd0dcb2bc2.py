#Qno 1

import pandas as pd

import numpy as np



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6], [1, 3]])
#Qno 2

from matplotlib import pyplot as plt

heights=[72,71,56,45,67,89,54,58,67,77,77,78,77,73,73,172,72,71,56,45,67,

         89,54,58,67,172,77,78,77,73,73,172,12,54,64,75,75,77,88,66,70,12,54,64,75,75,77,88,66,70]

def plot_his(heights):

    start=min(heights)-min(heights)%10

    end=max(heights)+10

    bins=list(range(start,end,5))

    plt.hist(heights,bins,histtype='bar',rwidth=0.5,color='#FF2400')

    plt.xlabel('heights in inches')

    plt.ylabel('No. of Students')

    plt.title("Heights chart")

    plt.show()

print("Abnormal Data")

plot_his(heights)

heights=list(filter(lambda x: not x==172 and not x==12, heights))

print("Correct Data")

plot_his(heights)
#Qno 3

import pandas as  pd

flights = pd.read_csv("../input/titanic/train_and_test2.csv")

flights[flights.isnull().any(axis=1)].head()