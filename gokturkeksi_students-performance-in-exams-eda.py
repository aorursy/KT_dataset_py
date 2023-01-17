import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This method takes csv file and translate a dataframe

data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
# This method takes first 5 row

data.head()
# This method takes randomly 5 row

data.sample(5)
data.info()
data.describe()
def bar_plot(variable):

    # get features

    var = data[variable]

    # get value counts

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (10,10))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{} : \n{}".format(variable,varValue))
category1 = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

for c in category1:

    bar_plot(c)
def hist_plot(variable):

    plt.figure(figsize = (10,10))

    plt.hist(data[variable],bins = 50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distrubition with Histogram".format(variable))

    plt.show()
numericVariable = ['math score', 'reading score', 'writing score']



for n in numericVariable:

    hist_plot(n)
data.columns
data[["gender","math score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "math score",ascending = False)
data[["gender","reading score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "reading score",ascending = False)
data[["gender","writing score"]].groupby(["gender"],as_index = True).mean().sort_values(by = "writing score",ascending = False)
data[["parental level of education","math score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "math score",ascending = False)
data[["parental level of education","reading score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "reading score",ascending = False)
data[["parental level of education","writing score"]].groupby(["parental level of education"],as_index = True).mean().sort_values(by = "writing score",ascending = False)
data[["race/ethnicity","math score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "math score",ascending = False)
data[["race/ethnicity","reading score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "reading score",ascending = False)
data[["race/ethnicity","writing score"]].groupby(["race/ethnicity"],as_index = True).mean().sort_values(by = "writing score",ascending = False)
data.columns
data[["test preparation course","math score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "math score",ascending = False)
data[["test preparation course","reading score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "reading score",ascending = False)
data[["test preparation course","writing score"]].groupby(["test preparation course"],as_index = True).mean().sort_values(by = "writing score",ascending = False)
def detect_outliers(df,features):

    outlier_indeces = []

    for c in features:

        Q1 = np.percentile(df[c],25)

        Q3 = np.percentile(df[c],75)

        IQR = Q3 - Q1

        outlier_step = IQR * 1.5

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        outlier_indeces.extend(outlier_list_col)

    outlier_indeces = Counter(outlier_indeces)

    multipler_outliers = list(k for k,v in outlier_indeces.items() if v > 2)

    return multipler_outliers
data.loc[detect_outliers(data,["math score","reading score","writing score"])]
# This method drops our outliers

data = data.drop(detect_outliers(data,['math score', 'reading score','writing score']),axis = 0).reset_index(drop = True)
data.loc[detect_outliers(data,["math score","reading score","writing score"])]
data.columns[data.isnull().any()]
data.info()
data.isnull().any()
data.isnull().sum()