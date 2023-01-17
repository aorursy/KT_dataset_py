# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/student-grade-prediction/student-mat.csv")

data['G_avg']= round((data['G1']+data['G2']+data['G3'])/3, 2)
#Distribution of Grades

plt.hist(data['G_avg'], bins=20, range=[0,20])

plt.title('Distribution of grades of students')

plt.xlabel('Grades')

plt.ylabel('Count')

plt.show()
data.boxplot(column=["G_avg"], by=["sex"])

data.boxplot(column=["G_avg"], by=["reason"])

data.boxplot(column=["G_avg"], by=["address"])

data.boxplot(column=["G_avg"], by=["famsize"])

data.boxplot(column=["G_avg"], by=["Pstatus"])

# data.boxplot(column=["G_avg"], by=["Medu"])

# data.boxplot(column=["G_avg"], by=["Fedu"])

data.boxplot(column=["G_avg"], by=["Mjob"])

data.boxplot(column=["G_avg"], by=["Fjob"])

data.boxplot(column=["G_avg"], by=["guardian"])

data.boxplot(column=["G_avg"], by=["paid"])

data.boxplot(column=["G_avg"], by=["famsup"])

data.boxplot(column=["G_avg"], by=["schoolsup"])

data.boxplot(column=["G_avg"], by=["activities"])

data.boxplot(column=["G_avg"], by=["nursery"])

data.boxplot(column=["G_avg"], by=["higher"]) #want uni

data.boxplot(column=["G_avg"], by=["internet"])

data.boxplot(column=["G_avg"], by=["romantic"])
data.describe()
#95 confidence interval

sns.lmplot('age','G_avg', data)

sns.lmplot('Medu','G_avg', data)

sns.lmplot('Fedu','G_avg', data)

sns.lmplot('traveltime','G_avg', data)

sns.lmplot('studytime','G_avg', data)

sns.lmplot('failures','G_avg', data)

sns.lmplot('famrel','G_avg', data)

sns.lmplot('freetime','G_avg', data)

sns.lmplot('goout','G_avg', data)

sns.lmplot('health','G_avg', data)

sns.lmplot('absences','G_avg', data)

sns.lmplot('Dalc','G_avg', data)

sns.lmplot('Walc','G_avg', data)



slope, intercept, r_value, p_value, std_err = stats.linregress(data['age'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['Medu'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['Fedu'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['traveltime'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['studytime'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['failures'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['famrel'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['freetime'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['goout'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['health'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['absences'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['Dalc'], data['G_avg'])

print(r_value)

slope, intercept, r_value, p_value, std_err = stats.linregress(data['Walc'], data['G_avg'])