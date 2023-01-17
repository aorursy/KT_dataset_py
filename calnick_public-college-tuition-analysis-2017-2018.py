# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 4. Write code to read the data - only read the needed columns: CONTROL, TUITIONFEE_IN



column_list = ["CONTROL", "TUITIONFEE_IN"]



df= pd.read_csv(r"/kaggle/input/college-tuition-data-20172018/MERGED2017_18_PP.csv", usecols=column_list)
# 5. Filter the data so you have only public institutions

# 1 = public



df = df[df["CONTROL"] == 1]
# 6. Drop the colleges that either have zero tuition or have missing tuition. The remaining colleges constitute the population.



df = df[df["TUITIONFEE_IN"] != 0]

df.dropna()

df.info()
df.head()
# 7. Get a random sample of 100 colleges from the population. 
# Random sample of colleges

SAMPLE_SIZE = 100         # This variable will be used through out the rest of cells



df_sample = df.sample(SAMPLE_SIZE)
# 8. Calculate the sample mean and sample standard error.

sample_mean = df_sample["TUITIONFEE_IN"].mean()

sample_mean                   
sample_std = np.std(df_sample["TUITIONFEE_IN"], ddof=1)

round(sample_std, 2)                    # Calculate sample Standard Deviation first
# Since we assume we don't know the population standard deviation, we use sample standard deviation as an estimate



std_err = sample_std / math.sqrt(SAMPLE_SIZE)       # standard error

std_err
# 9. Calculate the confidence interval of the mean estimate at 68%, 95%. and 99.7%
# Calculate 68% Confidence Interval (CI) - one standard error from the population mean

# 68% chances the population mean is within the sample_mean (+ or -) the standard error (SE)



LCL_68 = sample_mean -  std_err

UCL_68 = sample_mean +  std_err



print("Lower confidence limit at 68% confidence level = ", round(LCL_68,2))

print("Upper confidence limit at 68% confidence level = ", round(UCL_68,2))
# Calculate 95% Confidence Interval (CI) - one standard error from the population mean

# 90% chances the population mean is within the sample_mean + or - 2 * the standard error (SE)



LCL_95 = sample_mean -  2 * std_err

UCL_95 = sample_mean +  2 * std_err

print("Lower confidence limit at 95% confidence level = ", round(LCL_95,2))

print("Upper confidence limit at 95% confidence level = ", round(UCL_95,2))

# Calculate 99.7% Confidence Interval (CI) - one standard error from the population mean

# 99.7% chances the population mean is within the sample_mean + or - 3 * the standard error (SE)



LCL_997 = sample_mean -  3 * std_err

UCL_997 = sample_mean +  3 * std_err

print("Lower confidence limit at 99.7% confidence level = ", round(LCL_997,2))

print("Upper confidence limit at 99.7% confidence level = ", round(UCL_997,2))
# 10. Calculate the population mean.



pop_mean = df["TUITIONFEE_IN"].mean()

pop_mean

# 11. Compare the population mean with the sample mean - display the difference



pop_mean - sample_mean
# 12. Check the confidence intervals and determine if the population mean within the confidence intervals calculated above.

# You don't need to write code, just check using your eyes.
