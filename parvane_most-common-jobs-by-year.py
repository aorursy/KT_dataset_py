import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import nltk as nl

input_file = pd.read_csv("../input/Salaries.csv",  low_memory=False)
input_file.info()
input_file['JobTitle2'] = input_file['JobTitle'].str.lower()
input_file1  = input_file[input_file['Year'] == 2011]
unique_jobs1 = input_file1['JobTitle2']
data1 = unique_jobs1.value_counts()[:25]
data1.plot(kind="bar", color = "orange")

input_file2  = input_file[input_file['Year'] == 2012]
unique_jobs2 = input_file2['JobTitle2']
data2 = unique_jobs2.value_counts()[:25]
data2.plot(kind="bar", color = "orange")

input_file3  = input_file[input_file['Year'] == 2013]
unique_jobs3 = input_file3['JobTitle2']
data3 = unique_jobs3.value_counts()[:25]
data3.plot(kind="bar", color = "orange")
input_file4  = input_file[input_file['Year'] == 2014]
unique_jobs4 = input_file4['JobTitle2']
data4 = unique_jobs4.value_counts()[:25]
data4.plot(kind="bar", color = "orange")
