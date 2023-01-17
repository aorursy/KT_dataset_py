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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv",index_col=0)
data

data.info()

data.describe()
data[data.isnull().any(axis=1)]
sns.set(color_codes=True)
data["Salary_min"]= data["Salary Estimate"].str.split("-").apply(lambda x:x[0])


data["Salary_min"]= data["Salary_min"].apply(lambda x:x[1:-1])
data["Salary_max"] = data["Salary Estimate"].str.split("-").apply(lambda x:x[1])
data["Salary_max"] = data["Salary_max"].str.split("K").apply(lambda x:x[0])
data["Salary_max"]= data["Salary_max"].apply(lambda x:x[1:])
data["Revenue_l"] = data["Revenue"].str.split("to").apply(lambda x:x[0])
data["Revenue_l"] = data["Revenue_l"].apply(lambda x:x[1:])
data["Revenue_u"] = data["Revenue"].str.split("to").apply(lambda x:x[-1])
#data["Revenue_u"] = data["Revenue_u"].str.split("million").apply(lambda x:x[0])
#data["Revenue_u"] = data["Revenue_u"].apply(lambda x:x[1:])
data.head(5)

data["Salary_min"]= pd.to_numeric(data["Salary_min"])
data["Salary_max"]=pd.to_numeric(data["Salary_max"])
data_valrat= data[data["Rating"]>0]
sns.heatmap(data_valrat.corr(),annot=True);
#sns.pairplot(data_valrat);
data.sort_values(by="Salary_max",ascending=False)
Maxsal= data[data["Salary_max"]==190][["Job Title","Company Name","Rating","Location"]]
#Maxsal[["Job Title","Company Name"]]
Maxsal.sort_values(by=["Rating","Location"],ascending=(False,True))
Grouped = data.groupby(["Company Name","Job Title","Location"]).mean()
Grouped.sort_values(by=["Salary_max","Salary_min","Rating"],ascending=(False))
data.groupby("Company Name").mean().sort_values(by=["Salary_max","Salary_min","Rating"],ascending=(False))
data.groupby("Job Title").mean().sort_values(by=["Salary_max","Salary_min","Rating"],ascending=(False))
data.groupby("Location").mean().sort_values(by=["Salary_max","Salary_min","Rating"],ascending=(False))