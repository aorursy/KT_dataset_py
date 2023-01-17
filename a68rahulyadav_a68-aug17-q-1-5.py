#Q1_program to gat the power of an array values element wise
import pandas as pd
data_f1 = pd.DataFrame({'A':[2,4,6,8,10],
                        'B':[3,6,9,12,15],
                        'C':[4,8,12,16,20]})
data_f1
#Q2_create and display a dataframe from a dictionary dataset with index
import numpy as np
country_data = {'name':['India','USA','Australia','Pakistan',
                        'Bangladesh','Italy','Germany','Cuba'],
                'year':[1947,1776,np.nan,1947,1971,
                        np.nan,np.nan,1898],
               'secular':['Yes','Yes','Yes','No',
                          'No','No','Yes','No'],
            'continent':['Asia','North America','Ocenia','Asia',
                         'Asia','Europe','Europe','North America']}
labels = ['a','b','c','d','e','f','g','h']
data_f2 = pd.DataFrame(country_data,index=labels)
data_f2
#Q3_to get first three rows of a given dataset
print('first three rows of my dataset:')
print(data_f2.iloc[0:3])
#Q4_to select specified columns and rows for a given dataframe
print('my dataset with selected rows and columns:')
print(data_f2.iloc[[0,2,3,4,7],[0,1,3]])
#Q5_program to select the rows where the score is missing
print('rows of my dataset where the data(year) is missing i.e NaN:')
print(data_f2[data_f2['year'].isnull()])