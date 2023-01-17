# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import modules

import pandas as pd  # for dataframes

import numpy as np

import matplotlib.pyplot as plt # for plotting graphs

import seaborn as sns # for plotting graphs

#for reading comma-separated values (csv) files and creating a DataFrame

df = pd.read_csv("../input/loan-prediction-data/Loan_dataset.csv") 
#Returns the first 10 rows of the dataframe.

df.head(10)
df.boxplot(column='ApplicantIncome') # to check the distribution of data in a variable specially outliers.

plt.show()
#to show the distribution of variable "ApplicantIncome" 

df.ApplicantIncome.hist()
#Creates Histogram to show the distribution of variable "LoanAmount"

df.LoanAmount.hist() 
# plot for mean of each feature grouped by Gender

df.groupby(by = "Gender").mean().plot(kind="bar",color=['red', 'green', 'blue', 'orange', 'cyan'])
# plot for mean of each feature grouped by their marital status.

df.groupby(by = "Married").mean().plot(kind="bar",color=['green', 'blue', 'white', 'orange', 'cyan'])
# plot for mean of each feature grouped by education

df.groupby(by = "Education").mean().plot(kind="bar",color=['blue', 'green', 'pink', 'purple', 'cyan'])

# plot for mean of each feature grouped by area

df.groupby(by = "Property_Area").mean().plot(kind="bar",color=['red', 'green', 'blue', 'orange', 'cyan'])

# plot for mean of each feature for their Loan status

df.groupby(by = "Loan_Status").mean().plot(kind="bar",color=['red', 'green', 'blue', 'orange', 'cyan'])

# plot for mean of each feature grouped by Credit History

df.groupby(by = "Credit_History").mean().plot(kind="bar",color=['orange', 'green', 'blue', 'pink', 'cyan'])
# plot for mean of each feature grouped by their employment

df.groupby(by = "Self_Employed").mean().plot(kind="bar",color=['purple', 'green', 'blue', 'red', 'cyan'])
#Subplots using Seaborn

#This is how we can analyze the features one by one, to save time. It's a better option here to use Seaborn library and plot all the graphs in a single run using subplots.

features=['Married','Gender','Dependents','Education','Self_Employed','Credit_History','Loan_Status','Property_Area']

fig=plt.subplots(figsize=(10,15))

for i, j in enumerate(features):

    plt.subplot(4, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data = df)

    plt.xticks(rotation=90)

    plt.title("Count of Customers")  

  
#Subplots using Seaborn

#This is how we can analyze the features one by one, to save time. It's a better option here to use Seaborn library and plot all the graphs in a single run using subplots.

fig=plt.subplots(figsize=(10,15))

for i, j in enumerate(features):

    plt.subplot(4, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data =df, hue='Loan_Status')

    plt.xticks(rotation=90)

    plt.title("Count of Customers with respect to their Loan Status")    
corr = df.corr() #to find the pairwise correlation between variables of dataframe

print(corr)
import statsmodels.api as sm

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()