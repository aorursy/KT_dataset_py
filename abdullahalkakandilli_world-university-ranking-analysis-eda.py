# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
school_cdf = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
school_cdf.head()
school_cdf.columns
school_cdf.describe()
school_cdf.info()
def bar_plot(variable):
    """
    
        input: variable ex: teaching
        output: bar plot & value count
    """
    #get feature
    var = school_cdf[variable]
    #count number of value/sample
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)    
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable, varValue))
cat1 = ["teaching", "research", "citations", "student_staff_ratio", "international_students", "total_score"]
for n in cat1:
    bar_plot(n)
def detect_outliers(df, features):
    outlier_indices = []
    
    for n in features:
        # 1st quartile
        Q1 = np.percentile(df[n], 25)
        
        # 3rd quartile
        Q3 = np.percentile(df[n], 75)
    
        # IQR
        IQR = Q3 - Q1
        
        # Outlier Step
        outlier_step = IQR * 1.5
        
        # detect outlier and their indeces
        outlier_list_col = df[(df[n] < Q1 - outlier_step) | (df[n] > Q3 + outlier_step)].index
        
    
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
#float64(4): teaching, research, citations and student_staff_ratio
school_cdf.loc[detect_outliers(school_cdf, ["teaching", "research", "citations", "student_staff_ratio"])]
#there is no outliers for this dataset
school_cdf_len = len(school_cdf)

school_cdf.info()
school_cdf.columns[school_cdf.isnull().any()]
school_cdf.isnull().sum()
school_cdf[school_cdf["student_staff_ratio"].isnull()]
print(school_cdf["student_staff_ratio"].mean())
school_cdf["student_staff_ratio"] = school_cdf["student_staff_ratio"].fillna("18.4")
school_cdf[school_cdf["student_staff_ratio"].isnull()]
#work properly
# drop country
school_cdf.drop(columns=["country","female_male_ratio", "world_rank"], inplace = True)
school_cdf.head()
#string to numeric
str_col = school_cdf.select_dtypes(["object"]).columns
school_cdf[str_col] = school_cdf[str_col].replace('-', 0)
school_cdf["num_students"] = school_cdf["num_students"].str.replace(',', '')
school_cdf["international_students"] = school_cdf["international_students"].str.replace("%", "")

school_cdf[str_col] = school_cdf[str_col].apply(pd.to_numeric, errors="coerce", axis = 1)

school_cdf["international_students"] = school_cdf["international_students"] / 100
school_cdf.isna().sum()
school_cdf.drop(columns=["university_name"], inplace = True)
school_cdf.isna().sum()
school_cdf.dropna(inplace=True)

#I will use index as rank of universities

school_cdf.index = np.arange(1, len(school_cdf)+1)
print(school_cdf.dtypes)
print(school_cdf.isna().sum())
school_cdf.head(10)

