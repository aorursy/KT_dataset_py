# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")



import seaborn as sns

from collections import Counter

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
data.columns
data.head()
data.describe()
data.info()
def bar_plot(variable):

    """

    

    input: variable ex:"race"

    output: bar plot & value count

    """

    #get feature

    var=data[variable]

    #count number of categorical variable(value/sample)

    varValue=var.value_counts()

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index,varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))
category1=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

for c in category1:

    bar_plot(c)
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(data[variable])

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericvar=["math score","reading score","writing score"]

for n in  numericvar:

    plot_hist(n)
data[["test preparation course","math score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="math score",ascending=False)
data[["test preparation course","reading score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["test preparation course","writing score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="writing score",ascending=False)
# Creating average column by using math, writing and reading scores

data["Average"]=[(data["math score"][a]+data["writing score"][a]+data["reading score"][a])/3 for a in range(len(data)) ]
data.head()
data[["test preparation course","Average"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="Average",ascending=False)
data[["race/ethnicity","math score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="math score",ascending=False)
data[["race/ethnicity","reading score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["race/ethnicity","writing score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="writing score",ascending=False)
data[["race/ethnicity","Average"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="Average",ascending=False)
data[["gender","math score"]].groupby(["gender"],as_index=False).mean().sort_values(by="math score",ascending=False)
data[["gender","reading score"]].groupby(["gender"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["gender","writing score"]].groupby(["gender"],as_index=False).mean().sort_values(by="writing score",ascending=False)
data[["gender","Average"]].groupby(["gender"],as_index=False).mean().sort_values(by="Average",ascending=False)
data[["parental level of education","math score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="math score",ascending=False)
data[["parental level of education","reading score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["parental level of education","writing score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="writing score",ascending=False)
data[["parental level of education","Average"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="Average",ascending=False)
data[["lunch","math score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="math score",ascending=False)
data[["lunch","reading score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["lunch","reading score"]].groupby(["lunch"],as_index=False).mean().sort_values(by="reading score",ascending=False)
data[["lunch","Average"]].groupby(["lunch"],as_index=False).mean().sort_values(by="Average",ascending=False)
def detect_outliers(df,features):

    outlier_indices=[]

    for c in features:

        #1st quartile

        Q1=np.percentile(df[c],25)

        

        #3rd quartile

        Q3=np.percentile(df[c],75)

        

        #IQR

        IQR=Q3-Q1

        

        #Outliers Step

        outlier_step=IQR*1.5

        

        #Detect outlier and their indeces

        outlier_list_col=df[(df[c]<Q1-outlier_step)|(df[c]>Q3+outlier_step)].index

        

        #store indeces

        outlier_indices.extend(outlier_list_col)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)

    return multiple_outliers

        

                            

            
data.boxplot()
clean_data=data.drop(detect_outliers(data,["math score","writing score","reading score","Average"]),axis=0).reset_index(drop=True)
clean_data.boxplot() # We dropped some of the outliers.
clean_data.columns[clean_data.isnull().any()] #We do not have any column that contains missing values