# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
h1b_df = pd.read_csv("../input/h1b_kaggle.csv",header = 0)
h1b_df.shape
h1b_df.head()
h1b_df.dtypes
h1b_df.isnull().sum()
h1b_df = h1b_df.dropna(0, subset = ["CASE_STATUS", "FULL_TIME_POSITION", "PREVAILING_WAGE", "YEAR" ])

#just to check

h1b_df.isnull().sum()
#lets check year wise distribution

#1 count by year

year_count_series = h1b_df["YEAR"].value_counts()

year_count_df = year_count_series.to_frame()

year_count_df = year_count_df.reset_index()

#rename columns from (index, type) to (type , count)

year_count_df.columns = ["year", "counts"]

sns.barplot(y=year_count_df['counts'],x=year_count_df['year'])
#salary year wise

salary_year_mean_series = h1b_df.groupby("YEAR")["PREVAILING_WAGE"].mean()

salary_year_mean_df = salary_year_mean_series.to_frame()

salary_year_mean_df = salary_year_mean_df.reset_index()

#rename columns from (index, type) to (type , count)

salary_year_mean_df.columns = ["year", "salarymean"]

sns.barplot(y=salary_year_mean_df['salarymean'],x=salary_year_mean_df['year'])
#mean salary is decreasing year wise as number of applications are increasing

# median salary year wise

salary_year_median_series = h1b_df.groupby("YEAR")["PREVAILING_WAGE"].median()

salary_year_median_df = salary_year_median_series.to_frame()

salary_year_median_df = salary_year_median_df.reset_index()

#rename columns from (index, type) to (type , count)

salary_year_median_df.columns = ["year", "salarymedian"]

sns.barplot(y=salary_year_median_df['salarymedian'],x=salary_year_median_df['year'])

#median wage increasing with year
h1b_df["CASE_STATUS"].unique()
h1b_df.groupby("YEAR")["CASE_STATUS"].value_counts()

# used h1b_df.groupby("YEAR")["CASE_STATUS"].value_counts().sum() to verify
#Lets plot the above .

year_array = h1b_df["YEAR"].unique()

for year in year_array:

    year_index = h1b_df["YEAR"].values == year

    year_df = pd.DataFrame(h1b_df[year_index]["CASE_STATUS"].value_counts().reset_index())

    year_df.columns = ["STATUS", "COUNT"];

    plt.figure()

    bar_chart = sns.barplot(x = year_df["STATUS"], y = year_df["COUNT"] )

    bar_chart.set_xticklabels(bar_chart.get_xticklabels(), rotation=40, ha="right")

    bar_chart.set_title(year)