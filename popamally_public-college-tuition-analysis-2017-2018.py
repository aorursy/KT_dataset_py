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
# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import math

plt.rcParams["figure.figsize"] = (12,12) 
# Write code to read the data - only read the needed columns: CONTROL, TUITIONFEE_IN:



column_list=["CONTROL", "TUITIONFEE_IN"]



df=pd.read_csv("/kaggle/input/assignment-9-part-b-college-score-dataset/MERGED2017_18_PP.csv", usecols = column_list)

df

#Filter the data so you have only public institutions



public_df=df.loc[df["CONTROL"] == 1]

public_df
#Drop the colleges that either have zero tuition or have missing tuition. The remaining colleges constitute the population.



public_df = public_df[public_df["TUITIONFEE_IN"] != 0]

public_df.info()

#Drop the colleges that either have zero tuition or have missing tuition. The remaining colleges constitute the population.

public_df["TUITIONFEE_IN"].isna().sum()

public_df.dropna(inplace=True)

public_df.info()
#Get a random sample of 100 colleges from the population. 

sample_size =100

public_df_sample = public_df.sample(100)

public_df_sample
#Calculate the sample mean and sample standard error.

sample_mean = public_df_sample["TUITIONFEE_IN"].mean()

sample_std = np.std(public_df_sample["TUITIONFEE_IN"], ddof=1) #standard deviation

round(sample_std, 2)

print("The sample mean is", sample_mean)

print("The sample standard deviation is", sample_std)

std_error = sample_std / math.sqrt(sample_size) 

round(std_error,2)

print("The standard error is", sample_std)

# Calculate the confidence interval of the mean estimate at 68%, 95%. and 99.7%



#68%CI:



LCL_68 = sample_mean -  std_error

UCL_68 = sample_mean +  std_error



print("Lower confidence limit at 68% confidence level = ", round(LCL_68,2))

print("Upper confidence limit at 68% confidence level = ", round(UCL_68,2))



#95%CI:



LCL_95 = sample_mean -  (2 * std_error)

UCL_95 = sample_mean +  (2 * std_error)



print("Lower confidence limit at 95% confidence level = ", round(LCL_68,2))

print("Upper confidence limit at 95% confidence level = ", round(UCL_68,2))



#99.7%CI:



LCL_997 = sample_mean -  3 * std_error

UCL_997 = sample_mean +  3 * std_error



print("Lower confidence limit at 99.7% confidence level = ", round(LCL_997,2))

print("Upper confidence limit at 99.7% confidence level = ", round(UCL_997,2))

#Calculate the population mean.

population_mean = public_df_sample["TUITIONFEE_IN"].mean()

population_mean
#Compare the population mean with the sample mean - display the difference

difference = sample_mean - population_mean

difference
#Check the confidence intervals and determine if the population mean within the confidence intervals calculated above.



##the population mean is within the range of the 698%, 95%, and 99.7% confidence interval.  